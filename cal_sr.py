import os
import json
import argparse

def calculate_success_rate(log_dir):
    total_tasks = 0
    success_tasks = 0
    
    print(f"正在扫描目录: {log_dir} ...")

    # 遍历目录查找 _result.json
    for root, dirs, files in os.walk(log_dir):
        if "_result.json" in files:
            file_path = os.path.join(root, "_result.json")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # 检查 success 字段
                    if "success" in data:
                        total_tasks += 1
                        if data["success"] == 1:
                            success_tasks += 1
            except json.JSONDecodeError:
                print(f"警告: 无法解析 JSON 文件 {file_path}")
            except Exception as e:
                print(f"警告: 读取文件 {file_path} 时出错: {e}")

    if total_tasks == 0:
        print("未找到任何包含 'success' 字段的 _result.json 文件。")
        return

    sr = (success_tasks / total_tasks) * 100
    
    print("-" * 30)
    print(f"统计结果:")
    print(f"总任务数 (Total): {total_tasks}")
    print(f"成功数 (Success): {success_tasks}")
    print(f"失败数 (Fail): {total_tasks - success_tasks}")
    print(f"成功率 (SR): {sr:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # 默认路径设置为你当前查看的 logs 目录结构
    default_path = "logs/tests_unseen"
    
    parser = argparse.ArgumentParser(description="统计 DISCO 任务成功率")
    parser.add_argument("--dir", type=str, default=default_path, help="日志文件夹路径")
    
    args = parser.parse_args()
    
    if os.path.exists(args.dir):
        calculate_success_rate(args.dir)
    else:
        print(f"错误: 目录 '{args.dir}' 不存在。")



