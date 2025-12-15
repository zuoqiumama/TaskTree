import json
import os
import re
import sys
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.llm import LLM

ALLOWED_SUBTASKS = [
    "PickupObject",
    "ToggleObject",
    "PutObject",
    "PutPickObject",
    "SliceObject",
    "CoolObject",
    "HeatObject",
    "CleanObject",
    "FindSecond",
    "PickSecond",
]


def build_system_prompt() -> str:
    """System prompt for the planner LLM."""
    return (
        "You are the high-level planner for an embodied agent performing household tasks. "
        "You MUST respond with a JSON object containing the fields: "
        "task_description, structured_params, example_plan, candidate_plans. "
        "\n\n"
        "IMPORTANT GUIDELINES for candidate_plans:\n"
        "1. candidate_plans should contain 2-3 ALTERNATIVE plans that achieve the SAME goal as example_plan.\n"
        "2. Alternative plans should differ in STRATEGY (e.g., different intermediate objects or locations), "
        "NOT by adding unnecessary extra steps.\n"
        "3. Each plan must be MINIMAL - only include steps necessary to complete the task.\n"
        "4. Do NOT add steps after the goal is achieved (e.g., no PutObject after ToggleObject in look_at_obj_in_light tasks).\n"
        "\n"
        "Use ONLY these subtasks: " + ", ".join(ALLOWED_SUBTASKS) + ".\n"
        "Each plan is an ordered list of [Action, Arg] pairs.\n"
        "Output JSON only, no commentary or code fences."
    )


def build_user_instruction(
    task_description: str,
    structured_params: Dict[str, Any],
    example_plan: List[List[str]],
) -> str:
    """Compose the user instruction asking for alternative plans."""
    # Build task type specific hints
    task_type = structured_params.get("task_type", "")
    task_hints = ""
    
    if task_type == "look_at_obj_in_light":
        task_hints = (
            "\nTask Goal: Pick up an object and turn on a lamp while holding it. "
            "The task is complete when the lamp is toggled on while holding the object. "
            "Do NOT add any steps after ToggleObject."
        )
    elif task_type == "pick_and_place_simple":
        task_hints = (
            "\nTask Goal: Pick up an object and place it on a receptacle. "
            "The task is complete when the object is placed. "
            "Alternative plans might use different pickup strategies."
        )
    elif task_type == "pick_two_obj_and_place":
        task_hints = (
            "\nTask Goal: Pick up two objects of the same type and place them. "
            "The task requires FindSecond and PickSecond to handle the second object."
        )
    elif task_type == "pick_and_place_with_movable_recep":
        task_hints = (
            "\nTask Goal: Pick up an object, place it in a movable receptacle, "
            "then place the receptacle on a target location. Uses PutPickObject."
        )
    elif task_type == "pick_cool_then_place_in_recep":
        task_hints = (
            "\nTask Goal: Pick up an object, cool it in the fridge, then place it. "
            "CoolObject must use 'Fridge' as argument."
        )
    elif task_type == "pick_heat_then_place_in_recep":
        task_hints = (
            "\nTask Goal: Pick up an object, heat it in the microwave, then place it. "
            "HeatObject must use 'Microwave' as argument."
        )
    elif task_type == "pick_clean_then_place_in_recep":
        task_hints = (
            "\nTask Goal: Pick up an object, clean it at the sink, then place it. "
            "CleanObject must use 'SinkBasin' as argument."
        )
    
    payload = {
        "task_description": task_description,
        "structured_params": structured_params,
        "example_plan": example_plan,
        "candidate_plans": [],
    }
    
    return (
        f"Generate alternative plans for this task.{task_hints}\n\n"
        "Requirements:\n"
        "1. Each alternative plan should achieve the SAME goal as example_plan.\n"
        "2. Plans should be MINIMAL - no unnecessary steps.\n"
        "3. Alternatives can differ in object choices (e.g., Knife vs ButterKnife) "
        "or approach strategies, but must have the same end result.\n"
        "4. Output 2-3 candidate_plans that are valid alternatives.\n\n"
        "Input:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_llm_response(raw: str) -> Dict[str, Any]:
    """
    Parse the LLM response, tolerating optional code fences or extra text.
    Returns a normalized dict with candidate_plans filtered to valid steps.
    """
    text = raw.strip()

    # Try direct JSON first.
    try:
        data = json.loads(text)
        return _normalize_candidate_plans(data)
    except Exception:
        pass

    # Try to extract JSON from a fenced block.
    match = re.search(r"```(?:json)?\\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        block = match.group(1).strip()
        data = json.loads(block)
        return _normalize_candidate_plans(data)

    # Try to extract the first JSON object in the text.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        data = json.loads(text[start : end + 1])
        return _normalize_candidate_plans(data)

    raise ValueError("Unable to parse LLM response as JSON")


def _normalize_candidate_plans(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only well-formed [action, arg] pairs per plan."""
    plans = data.get("candidate_plans", [])
    valid_plans: List[List[List[str]]] = []
    for plan in plans:
        if not isinstance(plan, list):
            continue
        cleaned: List[List[str]] = []
        for step in plan:
            if isinstance(step, (list, tuple)) and len(step) == 2:
                action, arg = step
                cleaned.append([str(action), str(arg)])
        if cleaned:
            valid_plans.append(cleaned)
    data["candidate_plans"] = valid_plans
    return data


def validate_plan(plan: List[List[str]], example_plan: List[List[str]]) -> bool:
    """Check that a plan is non-empty, uses only allowed subtasks, and differs from example_plan."""
    if not plan:
        return False
    for step in plan:
        if len(step) != 2:
            return False
        action, arg = step
        if action not in ALLOWED_SUBTASKS:
            return False
        if arg is None or str(arg).strip() == "":
            return False
    # Filter out exact duplicate of the example plan.
    if plan == example_plan:
        return False
    return True


def filter_valid_plans(plans: List[List[List[str]]], example_plan: List[List[str]]) -> List[List[List[str]]]:
    """Return only valid plans."""
    return [p for p in plans if validate_plan(p, example_plan)]


class TaskNode:
    def __init__(self, action, arg):
        self.action = action
        self.arg = arg
        self.children = []  # List of TaskNode
        self.count = 1  # How many plans go through this node
        self.is_original_endpoint = False  # True if this is the last node of original plan

    def add_child(self, node):
        for child in self.children:
            if child.action == node.action and child.arg == node.arg:
                child.count += 1
                return child
        self.children.append(node)
        return node

    def to_dict(self):
        return {
            "action": self.action,
            "arg": self.arg,
            "children": [c.to_dict() for c in self.children],
            "count": self.count,
            "is_original_endpoint": self.is_original_endpoint
        }

def build_task_tree(plans: List[List[List[str]]], original_plan_index: int = 0) -> TaskNode:
    """
    Builds a task tree from a list of plans.
    Each plan is a list of [action, arg] pairs.
    
    The original plan (at original_plan_index) is guaranteed to be a single path
    where the last node is a leaf node (marked with is_original_endpoint=True).
    
    Candidate plans can share prefix with original plan, but if they have
    additional steps after the original plan's endpoint, those steps form
    a new branch from ROOT (not from the endpoint).
    """
    root = TaskNode("ROOT", None)
    
    if not plans:
        return root
    
    original_plan = plans[original_plan_index]
    original_path_length = len(original_plan)
    
    # First, add the original plan as a single path
    current_node = root
    original_path_nodes = [root]  # Track all nodes in original path
    for i, step in enumerate(original_plan):
        if len(step) == 2:
            action, arg = step
            new_node = TaskNode(action, arg)
            current_node = current_node.add_child(new_node)
            original_path_nodes.append(current_node)
            # Mark the last node as original endpoint
            if i == len(original_plan) - 1:
                current_node.is_original_endpoint = True
    
    # Then, add candidate plans
    for plan_idx, plan in enumerate(plans):
        if plan_idx == original_plan_index:
            continue  # Skip the original plan
        
        # Find where this plan diverges from the original plan
        diverge_point = 0
        for i, step in enumerate(plan):
            if i < original_path_length and len(step) == 2:
                orig_step = original_plan[i]
                if step[0] == orig_step[0] and step[1] == orig_step[1]:
                    diverge_point = i + 1
                else:
                    break
            else:
                break
        
        # If the plan is longer than original and shares the entire original path,
        # we need to handle the extra steps as a separate branch from ROOT
        if diverge_point >= original_path_length and len(plan) > original_path_length:
            # This candidate plan extends beyond original plan
            # Add the extra steps as a completely new path from ROOT
            # to avoid adding children to the original endpoint
            current_node = root
            for step in plan:
                if len(step) == 2:
                    action, arg = step
                    # Check if we should follow existing path or create new
                    existing = None
                    for child in current_node.children:
                        if child.action == action and child.arg == arg:
                            # Don't reuse the original endpoint node
                            if not child.is_original_endpoint:
                                existing = child
                            break
                    
                    if existing and not existing.is_original_endpoint:
                        existing.count += 1
                        current_node = existing
                    else:
                        new_node = TaskNode(action, arg)
                        current_node.children.append(new_node)
                        current_node = new_node
        else:
            # Normal case: plan diverges before or at the original endpoint
            # Start from the divergence point
            if diverge_point > 0:
                current_node = original_path_nodes[diverge_point]
            else:
                current_node = root
            
            for i, step in enumerate(plan):
                if i < diverge_point:
                    continue  # Skip shared prefix
                if len(step) == 2:
                    action, arg = step
                    new_node = TaskNode(action, arg)
                    current_node = current_node.add_child(new_node)
    
    return root

def print_tree(node: TaskNode, level=0):
    indent = "  " * level
    print(f"{indent}- {node.action} ({node.arg}) [count={node.count}]")
    for child in node.children:
        print_tree(child, level + 1)
