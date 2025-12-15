import numpy as np
import json
import pickle as pkl
import sys
import os
# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (假设 train/scene_gen/ 是两层深度，根据实际情况调整 '../..')
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# 将根目录加入系统路径
sys.path.append(project_root)
from utils import utils
from collections import defaultdict
import threading
from env.thor_env import ThorEnv
import gen.constants as constants
import argparse
import copy



def main(rank, args):
    env = ThorEnv(x_display=args.display)
    while True:
        lock.acquire()
        if len(all_scene_numbers) >0:
            scene_num = all_scene_numbers.pop()
        else:
            scene_num = None
        lock.release()
        
        if scene_num is None:
            break
        scene_name = ('FloorPlan%d') % scene_num
        env.reset(  
            scene_name,
            render_image=True,
            render_depth_image=False,
            render_class_image=False,
            render_object_image=False)
        
        # print('Rank %d Running Scene %d' % (rank,scene_num))
        
        agent_height = env.last_event.metadata['agent']['position']['y']
        env.step(dict(action='GetReachablePositions', gridSize=constants.AGENT_STEP_SIZE ))
        action_return = env.last_event.metadata['actionReturn']
        if action_return is None:
            print("ERROR: scene %d 'GetReachablePositions' returns None" % scene_num)
        
        valid_pts = []
        for pt in action_return:
            x = pt['x']
            z = pt['z']
            point_is_valid = True
            for h in [45, 0]:
                if not point_is_valid:
                    break
                
                for r in [0,90,180,270]:
                    if not point_is_valid:
                        break
                    action = {
                        'action': 'TeleportFull',
                        'x': x,
                        'y': agent_height,
                        'z': z,
                        'rotateOnTeleport': True,
                        'rotation': r,
                        'horizon': h
                    }
                    event = env.step(action)
                    if not event.metadata['lastActionSuccess']:
                        point_is_valid = False
                
                # action = {
                #     'action': 'TeleportFull',
                #     'x': x,
                #     'y': agent_height,
                #     'z': z,
                #     'rotateOnTeleport': True,
                #     'rotation': 0,
                #     'horizon': h
                # }
                # event = env.step(action)
                # if not event.metadata['lastActionSuccess']:
                #     point_is_valid = False
                
                # for i in range(3):
                #     action = {'action': 'RotateLeft'}
                #     event = env.step(action)
                #     if not event.metadata['lastActionSuccess']:
                #         point_is_valid = False
                        
                        
                        
            if point_is_valid:
                valid_pts.append([x,z])
        official_nav_pts = np.load(f'./gen/layouts/FloorPlan{scene_num}-layout.npy')
        print(f"scene {scene_name}. reachable {len(action_return)}. check_valid {len(valid_pts)}. AlfredRelease {len(official_nav_pts)}")
        save_path = f'./data/scene_parse/nav/{scene_name}.pkl'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path,'wb') as f:
            pkl.dump(valid_pts,f)

if __name__ == '__main__':
    # all_scene_numbers = [1]
    lock = threading.Lock()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_proc',default=1,type=int)
    parser.add_argument('--display',default=0,type=str)
    parser.add_argument('--scene', default=None,type=int,nargs='*')
    args = parser.parse_args()
    
    if args.scene is None:    
        all_scene_numbers = sorted(constants.TRAIN_SCENE_NUMBERS + constants.TEST_SCENE_NUMBERS, reverse=True)
    else:
        all_scene_numbers = copy.deepcopy(args.scene)
        all_scene_numbers = all_scene_numbers[::-1]
    
    
    if args.n_proc == 1:
        main(0, args)
    else:
        threads = []
        for n in range(args.n_proc):
            thread = threading.Thread(target=main,args=(n,args,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
