# DISCO AI Agent Instructions

## Project Overview
DISCO (Differentiable Scene Semantics and Dual-level Control) is an embodied AI agent for the ALFRED benchmark. It navigates and interacts in AI2-THOR environments using a differentiable semantic mapping module and a hierarchical planner.

## Architecture & Core Components

### Agent Hierarchy
- **`agent/base.py`**: Abstract base class handling task lifecycle (`launch`, `reset_task`), logging, and evaluation metrics (`evaluate_alfred_task`).
- **`agent/perceiver.py`** (Differentiable SLAM):
  - **Core Logic**: `perceive(learn=True)` predicts depth/segmentation/affordance, projects to 3D (`slam()`), and updates `self.scene_map` via backpropagation (`optimize_scene()`).
  - **State**: `self.scene_map` (learnable tensor), `self.pose` (discrete grid), `self.object_queries`, `self.affordance_queries`.
  - **Mapping**: Projects 2D observations to a 3D point cloud, then to a 2D grid map (`self.map_size` x `self.map_size`).
- **`agent/disco.py`** (High-Level Policy):
  - **Planning**: `make_plan()` generates a fixed sequence of subgoals (e.g., `['PickupObject', 'Apple']`) based on task type. It does NOT use the PDDL planner at inference time.
  - **Execution**: `run()` iterates through subgoals.
  - **Subgoals**: Methods like `pickup()`, `put()`, `toggle()` follow the pattern: `common()` (find & nav) -> `finetune_action()` (align) -> `va_interact()` (execute).
  - **Fine-Tuning**: `finetune_action()` uses `FinePolicyNet` (`models/resnet_policy.py`) to adjust rotation/horizon for precise interaction.

### Environment
- **`env/thor_env.py`**: Wrapper for `ai2thor` (v2.1.0).
  - **Headless**: Manages X server display.
  - **Interaction**: `va_interact(action, interact_mask)` executes actions. It uses the `interact_mask` to identify the target object in THOR by calculating IoU with ground truth instance masks.
  - **Smooth Navigation**: `smooth_nav=True` in `step()` interpolates frames for visualization.

### Perception Models (`models/`)
- **`affunet.py`**: Affordance segmentation (U-Net).
- **`depthunet.py`**: Depth estimation.
- **`resnet_policy.py`**: `FinePolicyNet` for fine-grained view adjustment before interaction.

## Critical Workflows

### 1. Headless Execution (MANDATORY)
On headless servers, you **MUST** start the X server separately before running the agent.
```bash
# Terminal 1: Start X server
python startx.py --gpu 0 --display 0

# Terminal 2: Run Agent
python run.py --n_proc 1 --x_display 0 ...
```

### 2. Inference & Evaluation
- **Entry Point**: `run.py` (uses `torch.multiprocessing`).
- **Command**:
  ```bash
  python run.py --n_proc 4 --split valid_seen --x_display 0 --name my_experiment
  ```
- **Evaluation**: `evaluate.py` calculates metrics from logs.
- **Arguments**:
  - `--n_proc`: Number of processes (and GPUs if available).
  - `--split`: `valid_seen`, `valid_unseen`, `tests_seen`, `tests_unseen`.
  - `--x_display`: Must match the display number from `startx.py`.

### 3. Debugging
- **Logs**: `logs/[EXP_NAME]/[TASK_ID]/`.
- **Visualization**: Enable with `--vis` or `args.vis = True`.
  - Outputs: `step_XXXXX_rgb.png`, `_seg.png`, `_navmap.png`.
  - Use `agent.log(msg)` for text logging.

## Coding Conventions & Patterns

### Coordinate Systems (CRITICAL)
- **Grid Map**: `self.map_size` x `self.map_size` (default 300x300).
- **Pose**: `[x, z, rotation, horizon]`.
  - `x, z`: Grid indices (integers).
  - `rotation`: Degrees (0, 90, 180, 270).
  - `horizon`: Degrees (0, 15, 30, 45).
- **Conversion**: `slam()` handles the projection from camera coordinates to the grid map. Be careful when converting between AI2-THOR continuous coordinates and the discrete grid map.

### Differentiable Mapping
- **Online Learning**: The map is updated *during inference*.
- **Optimization**: `self.optimizer` updates `self.scene_map` to minimize projection error between the map and current 2D observations.
- **Queries**: `self.object_queries` and `self.affordance_queries` are learnable embeddings used to decode the map.

### Navigation
- **Navigable Map**: Derived from `self.affordance_map` (threshold > 0.6) and `self.collision_map`.
- **Pathfinding**: BFS on the grid map (`plan_step_to_waypoint`).
- **Exploration**: `sample_random_waypoint` or `sample_object_waypoint` (using dilation) when target is not visible.

### Interaction
- **Mask-Based**: `va_interact(action, mask)` uses segmentation masks to identify target objects in THOR, not just object IDs.
- **Fine-Tuning**: Before interacting, `finetune_action()` adjusts rotation/horizon to center the target object using `FinePolicyNet`.

## Common Pitfalls
- **AI2-THOR Version**: Strictly requires `ai2thor==2.1.0`.
- **Dependency Stack (CRITICAL)**: This legacy project requires a specific stack from ~2020. Newer versions will cause ImportErrors (e.g., `cannot import name 'escape' from 'jinja2'`).
  - `ai2thor==2.1.0`
  - `werkzeug==0.16.1`
  - `flask==1.1.2`
  - `Jinja2==2.11.3`
  - `MarkupSafe==1.1.1`
  - `itsdangerous==1.1.0`
- **Network/Proxy (CRITICAL)**: AI2-THOR communicates via local HTTP. You **MUST** unset proxy variables (`unset http_proxy https_proxy`) before running `run.py`. Failure to do so causes `SocketException` in Unity and Timeouts in Python.

