import json
import os
import glob
import sys
from collections import Counter

# 尝试导入 tqdm 显示进度条，如果没有则定义一个简单的占位符
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

# Acc 0.9
def infer_task_type_rule_based(task_input):
    """
    根据自然语言任务描述推断 ALFRED 的 7 种任务类型。
    支持传入单个描述字符串，或描述列表（进行投票）。
    """
    if isinstance(task_input, list):
        # 投票机制：对每个描述进行推断
        votes = [_infer_single_desc(desc) for desc in task_input]
        if not votes:
            return "pick_and_place_simple" # Default fallback
        # 返回出现次数最多的类型
        return Counter(votes).most_common(1)[0][0]
    else:
        return _infer_single_desc(task_input)

def _infer_single_desc(task_desc):
    """
    单条描述的推断逻辑
    """
    desc = task_desc.lower()
    
    # 1. Look At (Examine)
    if "examine" in desc or "look at" in desc or "lamp" in desc or "light" in desc:
        return "look_at_obj_in_light"
    
    # 2. Pick Two (Multiple objects)
    # 关键词: two, 2, all, both, gather, pair
    # 放在 Clean/Heat/Cool 之前，因为 "Pick two apples and put in sink" 是 Pick Two 任务
    if "two" in desc or " 2 " in desc or "all" in desc or "both" in desc or "gather" in desc or "pair" in desc:
        return "pick_two_obj_and_place"
    
    # 3. Clean (Wash)
    # 关键词: clean, wash, rinse, scrub
    # 移除了 "sink"，因为 "put apple in sink" 可能是 Simple Place
    if "clean" in desc or "wash" in desc or "rinse" in desc or "scrub" in desc:
        return "pick_clean_then_place_in_recep"
    
    # 4. Heat (Cook)
    # 关键词: heat, cook, warm, hot, burn, toast, microwaved
    # 移除了 "microwave" 名词，因为 "put apple in microwave" 可能是 Simple Place
    # 但保留 "microwaved" (形容词)
    if "heat" in desc or "cook" in desc or "warm" in desc or "hot" in desc or "burn" in desc or "toast" in desc or "microwaved" in desc:
        return "pick_heat_then_place_in_recep"
    
    # 5. Cool (Chill)
    # 关键词: cool, chill, freeze, cold, fridge
    # "fridge" 通常暗示 Cool 任务，不像 microwave 那么容易混淆
    if "cool" in desc or "chill" in desc or "freeze" in desc or "cold" in desc or "fridge" in desc:
        return "pick_cool_then_place_in_recep"
    
    # 6 & 7. Pick & Place (Simple vs Movable)
    else:
        # 区分 Simple 和 MovableRecep 的关键在于目标容器是否可移动。
        movable_receps = [
            "box", "mug", "bowl", "plate", "tray", "basket", 
            "bucket", "cup", "pan", "pot", "kettle", "watering can",
            "pitcher", "container", "bin", "can", "cart", "trash can"
        ]
        
        # 检查描述中是否包含将物体放入这些可移动容器
        is_movable = any(r in desc for r in movable_receps)
        
        if is_movable:
            return "pick_and_place_with_movable_recep"
        else:
            return "pick_and_place_simple"

def evaluate_split(data_root, split_name):
    split_path = os.path.join(data_root, split_name)
    if not os.path.exists(split_path):
        print(f"Error: Path not found: {split_path}")
        return 0, 0, []

    # 查找所有 traj_data.json
    # 使用递归搜索 (**)，因为 valid_seen 结构是 split/task/trial/traj_data.json (2层)
    search_pattern = os.path.join(split_path, "**", "traj_data.json")
    traj_files = glob.glob(search_pattern, recursive=True)
    
    if not traj_files:
        print(f"Warning: No json files found in {search_pattern}")
        return 0, 0, []

    correct = 0
    total = 0
    errors = []

    print(f"Evaluating {split_name} with {len(traj_files)} trials...")
    
    for json_file in tqdm(traj_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
            
        ground_truth_type = data['task_type']
        
        if 'turk_annotations' in data and 'anns' in data['turk_annotations']:
            anns = data['turk_annotations']['anns']
            
            # 修改：收集所有描述并进行投票推断
            task_descs = [ann['task_desc'] for ann in anns]
            predicted_type = infer_task_type_rule_based(task_descs)
            
            if predicted_type == ground_truth_type:
                correct += 1
            else:
                errors.append({
                    'desc': str(task_descs), # 记录所有描述以便分析
                    'gt': ground_truth_type,
                    'pred': predicted_type,
                    'file': json_file
                })
            total += 1
        else:
            pass
            
    accuracy = correct / total if total > 0 else 0
    print(f"Split: {split_name} | Accuracy: {accuracy:.2%} ({correct}/{total})")
    return correct, total, errors

if __name__ == "__main__":
    # 数据集路径
    DATA_ROOT = "/home/ucas03/home/ucas03/disco/DISCO/data/json_2.1.0"
    
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root directory '{DATA_ROOT}' does not exist.")
        print("Please make sure you are running this script from the project root.")
        sys.exit(1)
    
    splits = ["valid_seen", "valid_unseen"]
    
    total_correct = 0
    total_count = 0
    all_errors = []
    
    print("Starting Task Type Inference Evaluation...")
    print("==========================================")
    
    for split in splits:
        c, t, errs = evaluate_split(DATA_ROOT, split)
        total_correct += c
        total_count += t
        all_errors.extend(errs)
        print("-" * 40)
        
    overall_acc = total_correct / total_count if total_count > 0 else 0
    print(f"\nOverall Accuracy: {overall_acc:.2%} ({total_correct}/{total_count})")
    
    # 打印错误分析
    if all_errors:
        print("\n--- Error Analysis (Top 20 Examples) ---")
        shown_descs = set()
        count = 0
        for err in all_errors:
            if err['desc'] not in shown_descs:
                print(f"Desc: {err['desc']}")
                print(f"   GT:   {err['gt']}")
                print(f"   Pred: {err['pred']}")
                print()
                shown_descs.add(err['desc'])
                count += 1
                if count >= 20:
                    break