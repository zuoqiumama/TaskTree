import sys
import os
import json

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

from models.llm import LLM
from models.task_tree_llm import (
    build_user_instruction,
    build_system_prompt,
    parse_llm_response,
    filter_valid_plans,
    build_task_tree,
    print_tree,
)

if __name__ == "__main__":
    example_plan = [["PickupObject", "AnyKnife"],["SliceObject", "Potato"],["PutObject", "SinkBasin"],["PickupObject", "PotatoSliced"],["HeatObject", "Microwave"],["PutObject", "SinkBasin"],]

    structured_params = {
        "task_type": "pick_heat_then_place_in_recep",
        "mrecep_target": "Microwave",
        "object_target": "Potato",
        "parent_target": "SinkBasin",
        "sliced": True,
    }

    task_description = "Slice the potato, heat it, and place it in the sink."

    user_instruction = build_user_instruction(
        task_description=task_description,
        structured_params=structured_params,
        example_plan=example_plan,
    )

    system_prompt = build_system_prompt()
    api_key = "sk-xwucfzugonxtxwpuopkwufuverlbwvurulsvgwyrxqjrqjuq"

    print("=== System Prompt ===")
    print(system_prompt)
    print("\n=== User Instruction ===")
    print(user_instruction)

    if not api_key:
        print("\n[WARN] No API key found in SILICONFLOW_API_KEY or LLM_API_KEY. Skipping live LLM call.")
        raise SystemExit(0)

    llm = LLM(api_key=api_key, system_prompt=system_prompt)
    raw_response = llm.query(ins=user_instruction, enable_thinking=False)

    print("\n=== Raw LLM Response ===")
    print(raw_response)

    try:
        parsed = parse_llm_response(raw_response)
        valid = filter_valid_plans(parsed.get("candidate_plans", []), example_plan)
        parsed["candidate_plans"] = valid
        print("\n=== Valid Candidate Plans ===")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        if not valid:
            print("\n[WARN] No valid plans after validation.")
        
        # Build and print the tree
        print("\n=== Task Tree ===")
        all_plans = [example_plan] + valid
        root = build_task_tree(all_plans)
        print_tree(root)

    except Exception as exc:
        print(f"\n[ERROR] Failed to parse LLM response: {exc}")
