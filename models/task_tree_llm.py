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
        "You are the high-level planner for an embodied agent. "
        "You MUST respond with a JSON object containing the fields "
        "task_description, structured_params, scene_objects, example_plan, candidate_plans. "
        "candidate_plans must contain multiple alternative plans that differ from example_plan "
        "but still achieve the task. Use ONLY these subtasks: "
        + ", ".join(ALLOWED_SUBTASKS)
        + ". Each plan is an ordered list of [Action, Arg] pairs. "
        "Do not include commentary or code fences in the final output."
    )


def build_user_instruction(
    task_description: str,
    structured_params: Dict[str, Any],
    example_plan: List[List[str]],
) -> str:
    """Compose the user instruction asking for alternative plans."""
    payload = {
        "task_description": task_description,
        "structured_params": structured_params,
        "example_plan": example_plan,
        "candidate_plans": [],
    }
    return (
        "Given the input JSON, fill candidate_plans with at least 2-3 feasible alternatives "
        "that differ from example_plan. Output JSON only.\n\nInput:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
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
            "count": self.count
        }

def build_task_tree(plans: List[List[List[str]]]) -> TaskNode:
    """
    Builds a task tree from a list of plans.
    Each plan is a list of [action, arg] pairs.
    """
    root = TaskNode("ROOT", None)
    for plan in plans:
        current_node = root
        for step in plan:
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
