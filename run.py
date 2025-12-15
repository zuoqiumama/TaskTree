from utils import arguments, utils
import os
import torch.multiprocessing as mp
import importlib
from env import thor_env
import time
from agent.disco import Agent
import copy
import torch # 添加 torch 导入
import gc    # 添加 gc 导入

# for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
#     if k in os.environ:
#         print(f"Removing proxy env var: {k}")
#         del os.environ[k]

def work(args, tasks_queue, lock):
    n_tasks = tasks_queue.qsize()
    env = thor_env.ThorEnv(x_display = args.x_display)
    agent = Agent(args,env)
    
    while True:
        lock.acquire()
        n_tasks_remain = tasks_queue.qsize()
        if n_tasks_remain == 0:
            lock.release()
            break
        task_info = tasks_queue.get()
        lock.release()
        if 'tests' in args.split:
            if 'plan' not in task_info:
                # 修复：tests 集没有 plan，初始化为空列表会导致 env/tasks.py 中 get_num_subgoals 访问 high_pddl[-1] 时报错
                # 添加一个 dummy action 避免 IndexError
                # 修正结构：planner_action 必须是一个包含 'action' 键的字典，否则会报 TypeError: string indices must be integers
                task_info['plan'] = {'low_actions': [], 'high_pddl': [{'planner_action': {'action': 'NoOp'}}]}
            if 'pddl_params' not in task_info:
                # 修复：tests 集没有 pddl_params，会导致 env/tasks.py 中 get_target 报 KeyError
                # 初始化所有必要的 key 为默认值，使用空字符串防止后续 'in' 操作报 TypeError
                task_info['pddl_params'] = {
                    'object_target': '',
                    'parent_target': '',
                    'toggle_target': '',
                    'mrecep_target': '',
                    'object_sliced': False
                }
        # --- 新增：清理显存和垃圾回收 ---
        # 1. 强制 Python 进行垃圾回收，清除不再引用的对象
        gc.collect()

        task_start_time = time.time()
        result = agent.launch(task_info)
        task_end_time = time.time()
        task_run_time = task_end_time - task_start_time    

        task_index = task_info['task_index']
        goal_condition = '%d / %d' % (result['completed_goal_conditions'], result['total_goal_conditions'])
        print(f'{utils.timestr()} Rank {args.rank} {task_index} ({n_tasks - n_tasks_remain + 1}/{n_tasks}). RunTime {task_run_time:.2f}. Step {agent.steps}. FPS {agent.steps/task_run_time :.2f}. GC {goal_condition}')
        

        # -----------------------------

    env.stop()

def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)

    logging_dir = os.path.join('logs', args.name)
    os.makedirs(logging_dir, exist_ok=True)
    args.logging = logging_dir
    
    mp.set_start_method('spawn')
    manager = mp.Manager()
    lock = manager.Lock()

    tasks = utils.load_alfred_tasks(args)
    
    tasks_queue = manager.Queue()
    for t in tasks:
        tasks_queue.put(t)
    
    
    args.gpu = args.gpu * (args.n_proc // len(args.gpu) + 1)
    args.gpu = args.gpu[:args.n_proc]

    print(f'******************** Launch {len(tasks)} Tasks *** {args.n_proc} Processes *** Device {args.gpu}')
    if args.n_proc > 1:
        threads = []
        for rank in range(args.n_proc):
            args.rank = rank
            thread = mp.Process(
                target=work, 
                args= (args, tasks_queue, lock))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    else:
        # 修复：在单进程模式下也需要设置 rank，否则 work 函数中的 print 会报错
        args.rank = 0 
        work(args, tasks_queue, lock)

if __name__ == '__main__':
    main()