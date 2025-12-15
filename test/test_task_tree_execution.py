import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

from models.task_tree_llm import TaskNode

# Mocking Agent class since we can't instantiate the full environment
class MockAgent:
    def __init__(self):
        self.logs = []
        self.terminated = False
        self.task_tree = None
        
        # Mock action methods
        self.pickup = MagicMock(return_value=True)
        self.put = MagicMock(return_value=True)
        self.toggle = MagicMock(return_value=True)
        self.slice = MagicMock(return_value=True)
        self.cool = MagicMock(return_value=True)
        self.heat = MagicMock(return_value=True)
        self.clean = MagicMock(return_value=True)
        self.find_second = MagicMock(return_value=True)
        self.pick_second = MagicMock(return_value=True)
        self.putpick = MagicMock(return_value=True)
        
        self.look_down = MagicMock()

    def log(self, msg):
        self.logs.append(msg)
        print(f"[LOG] {msg}")

    def is_terminate(self):
        return self.terminated

    # Copy the execute_task_node method from disco.py (or import if possible, but here we mock the class)
    def execute_task_node(self, node):
        if node.action == "ROOT":
            # Try children
            for child in node.children:
                if self.execute_task_node(child):
                    return True
            return False
        
        self.log(f'============================================ Subgoal: {node.action} {node.arg}')
        if self.is_terminate():
            return True

        self.look_down()
        
        success = False
        if node.action == 'PickupObject':
            success = self.pickup(node.arg)
        elif node.action == 'ToggleObject':
            success = self.toggle(node.arg)
        elif node.action == 'PutObject':
            success = self.put(node.arg)
        elif node.action == 'PutPickObject':
            success = self.putpick(node.arg)
        elif node.action == 'SliceObject':
            success = self.slice(node.arg)
        elif node.action == 'CoolObject':
            success = self.cool(node.arg)
        elif node.action == 'HeatObject':
            success = self.heat(node.arg)
        elif node.action == 'CleanObject':
            success = self.clean(node.arg)
        elif node.action == 'FindSecond':
            success = self.find_second(node.arg)
        elif node.action == 'PickSecond':
            success = self.pick_second(node.arg)
        else:
            raise NotImplementedError(node.action)
            
        if not success:
            self.log(f"Subgoal {node.action} failed.")
            return False
            
        # If leaf, we are done
        if not node.children:
            return True
            
        # Try children
        for child in node.children:
            if self.execute_task_node(child):
                return True
        
        self.log(f"All children of {node.action} failed.")
        return False

class TestTaskTreeExecution(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent()

    def test_simple_linear_execution(self):
        # ROOT -> A -> B -> C
        root = TaskNode("ROOT", None)
        node_a = TaskNode("PickupObject", "Apple")
        node_b = TaskNode("PutObject", "Table")
        node_c = TaskNode("ToggleObject", "Light")
        
        root.add_child(node_a)
        node_a.add_child(node_b)
        node_b.add_child(node_c)
        
        self.agent.task_tree = root
        result = self.agent.execute_task_node(root)
        
        self.assertTrue(result)
        self.agent.pickup.assert_called_with("Apple")
        self.agent.put.assert_called_with("Table")
        self.agent.toggle.assert_called_with("Light")

    def test_backtracking_on_failure(self):
        # ROOT -> A -> B (Fails)
        #           -> C (Succeeds)
        root = TaskNode("ROOT", None)
        node_a = TaskNode("PickupObject", "Apple")
        node_b = TaskNode("PutObject", "Table") # Will fail
        node_c = TaskNode("PutObject", "Sink")  # Will succeed
        
        root.add_child(node_a)
        node_a.add_child(node_b)
        node_a.add_child(node_c)
        
        # Configure mock to fail on PutObject Table
        def put_side_effect(arg):
            if arg == "Table":
                return False
            return True
        self.agent.put.side_effect = put_side_effect
        
        self.agent.task_tree = root
        result = self.agent.execute_task_node(root)
        
        self.assertTrue(result)
        self.agent.pickup.assert_called_with("Apple")
        # Should have tried Table first
        # Then Sink
        self.assertEqual(self.agent.put.call_count, 2)
        
    def test_full_failure(self):
        # ROOT -> A (Fails)
        root = TaskNode("ROOT", None)
        node_a = TaskNode("PickupObject", "Apple")
        
        root.add_child(node_a)
        self.agent.pickup.return_value = False
        
        result = self.agent.execute_task_node(root)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
