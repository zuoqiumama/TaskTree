import json
import os
import re
import glob
from collections import defaultdict

def get_trial_id_from_log(log_path):
    """
    尝试从 log.txt 中提取 trial_id (例如 trial_T20190906_162633_239128)
    """
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
        # 常见的日志格式匹配，根据 agent/base.py 的记录方式调整
        # 假设日志中有类似 "Task: trial_..." 或 "Evaluate: trial_..." 的记录
        match = re.search(r'(trial_T\d{8}_\d{6}_\d{6})', content)
        if match:
            return match.group(1)
    return None

def load_split_data():
    """
    尝试加载标准的 ALFRED split 文件作为备用方案，以便通过索引查找 trial_id
    """
    split_path = 'data/splits/oct21.json'
    if os.path.exists(split_path):
        try:
            with open(split_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def clean_action(action_data):
    """
    清理动作数据，只保留提交所需的字段
    """
    if action_data is None:
        return None

    # 提交格式通常需要的字段
    allowed_keys = {
        'action', 
        'objectId', 
        'receptacleObjectId', 
        'forceAction', 
        'placeStationary',
        'x', 'y', 'z', 'rotation', 'horizon', 'standing' # Teleport 等动作可能需要
    }
    
    cleaned = {}
    for k, v in action_data.items():
        if k in allowed_keys:
            cleaned[k] = v
            
    return cleaned

def main():
    splits = ['tests_seen', 'tests_unseen']
    submission_data = {
        "tests_seen": [],
        "tests_unseen": []
    }
    
    # 加载 split 数据作为备用 ID 查找方式
    split_data = load_split_data()
    
    base_log_dir = 'logs'
    
    for split in splits:
        split_dir = os.path.join(base_log_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Directory {split_dir} not found.")
            continue
            
        print(f"Processing {split}...")
        
        # 获取所有任务文件夹 (假设格式为 TS00000, TU00000 等)
        task_folders = sorted(glob.glob(os.path.join(split_dir, '*')))
        
        for folder in task_folders:
            if not os.path.isdir(folder):
                continue
                
            folder_name = os.path.basename(folder)
            api_actions_path = os.path.join(folder, '_api_actions.json')
            log_path = os.path.join(folder, 'log.txt')
            
            if not os.path.exists(api_actions_path):
                # print(f"Skipping {folder_name}: _api_actions.json not found")
                continue
                
            # 1. 尝试获取 Trial ID
            trial_id = get_trial_id_from_log(log_path)
            
            # 2. 如果日志中没找到，尝试通过文件夹索引从 split 文件查找
            if not trial_id and split_data:
                try:
                    # 假设文件夹名以 TS 或 TU 开头，后面是数字索引
                    idx = int(re.search(r'\d+', folder_name).group())
                    if split in split_data and idx < len(split_data[split]):
                        trial_id = split_data[split][idx]['task']
                except Exception as e:
                    pass
            
            if not trial_id:
                print(f"Error: Could not determine trial_id for {folder_name}. Skipping.")
                continue
                
            # 3. 读取并清理动作
            try:
                with open(api_actions_path, 'r') as f:
                    actions = json.load(f)
                
                cleaned_actions = []
                if actions:
                    for a in actions:
                        cleaned = clean_action(a)
                        if cleaned is not None:
                            cleaned_actions.append(cleaned)
                
                # 4. 添加到结果列表
                # 格式: { "trial_ID": [actions] }
                submission_data[split].append({
                    trial_id: cleaned_actions
                })
                
            except Exception as e:
                print(f"Error processing {folder_name}: {e}")

    # 保存结果
    output_file = 'tests_submit.json'
    with open(output_file, 'w') as f:
        json.dump(submission_data, f, indent=4)
        
    print(f"Successfully generated {output_file}")
    print(f"Counts - Seen: {len(submission_data['tests_seen'])}, Unseen: {len(submission_data['tests_unseen'])}")

if __name__ == "__main__":
    main()