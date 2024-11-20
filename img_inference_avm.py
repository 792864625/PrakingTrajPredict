import argparse, os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import cv2
import numpy as np
import torch
from tqdm import tqdm
import glob
from avm import avm

from utils.config import get_inference_config_obj
from utils.fev2avm import FEV2AVM
from model_interface.inference_raw import ParkingInferenceRaw


def ros2show(point_x, point_y, center_x, center_y, center2rear):
    y = (center_y - point_x*100 + center2rear) 
    x = (center_x - point_y*100)

    return x,y

def normal2show(point_x, point_y, center_x, center_y, center2rear):
    #normal2ros
    x = (0.5 - point_y)*12.0
    y = (0.5 - point_x)*12.0
    #ros2show
    x,y = ros2show(x, y, center_x, center_y, center2rear)
    return x,y


if __name__ == "__main__":
    # import ipdb
    # ipdb.set_trace()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--inference_config_path', default="./config/inference_real.yaml", type=str)
    arg_parser.add_argument('--tasks_path', default="./e2e_dataset_gw/best_all/val/demo_bag", type=str)
    arg_parser.add_argument('--out_dir', default="./demo_output/test", type=str)
    args = arg_parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)


    inference_cfg = get_inference_config_obj(args.inference_config_path)

    parking_model = ParkingInferenceRaw(inference_cfg)
    # task test
    images_tag = parking_model.images_tag
    task_paths = [os.path.join(args.tasks_path, task_) for task_ in os.listdir(args.tasks_path)]

    avm_interface = avm()
    for task_path in task_paths:

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(os.path.join(args.out_dir, task_path.split('/')[-1] + '.mp4'), fourcc, 30, (avm_interface.bev_width, avm_interface.bev_height))
        print(f'frame id : {task_path}')
        frame_num = len(glob.glob(os.path.join(task_path, 'trajectory/*.npy')))
        for frame_idx in tqdm(range(frame_num), desc = os.path.basename(task_path)):
            file_prefix = f"{str(frame_idx).zfill(4)}"

            # inference input
            img_path_dict = {}
            img_fish_path_dict = {}
            for tag in images_tag:
                img_path_dict[tag] = os.path.join(task_path, tag, file_prefix + '.png')
                img_fish_path_dict[tag] = os.path.join(task_path, tag+"_fish", file_prefix + '.png')
            traj_file = os.path.join(task_path, 'trajectory', file_prefix + '.npy')
            traj_data = np.load(traj_file, allow_pickle=True).item()
            target_point = traj_data['target']
            traj_gt = traj_data['trajectory'].reshape(-1, 2)
            out_traj = parking_model.predict(img_path_dict, None, target_point, is_raw=True)

            bev =avm_interface.apply(img_path_dict)
            target_point = traj_data['target']
            x,y = ros2show(target_point[0], target_point[1], avm_interface.bev_center_x, avm_interface.bev_center_y, avm_interface.center2rear)
            cv2.circle(bev, (int(x), int(y)), 1, (255,255,255),30)


            pred_pt = (0,0)
            pred_prev_pt = (0,0)
            for i in range(len(out_traj)):
                x,y = normal2show(out_traj[i][0], out_traj[i][1], avm_interface.bev_center_x, avm_interface.bev_center_y, avm_interface.center2rear)
                cv2.circle(bev, (int(x), int(y)), 1, (255,255,255),1)
                x,y = normal2show(out_traj[i][0], out_traj[i][1], avm_interface.bev_center_x, avm_interface.bev_center_y, avm_interface.center2rear)
                pred_pt = (int(x),int(y))
                if i > 0:    
                    cv2.line(bev, pred_prev_pt, pred_pt, (255,255,255), 1) 
                pred_prev_pt = pred_pt

            

            gt_pt = (0,0)
            gt_prev_pt = (0,0)
            for i in range(len(traj_gt)):
                x,y = ros2show(traj_gt[i][0], traj_gt[i][1], avm_interface.bev_center_x, avm_interface.bev_center_y, avm_interface.center2rear)
                cv2.circle(bev, (int(x), int(y)), 1, (0,255,0),1)

                x,y = ros2show(traj_gt[i][0], traj_gt[i][1], avm_interface.bev_center_x, avm_interface.bev_center_y, avm_interface.center2rear)
                gt_pt = (int(x),int(y))
                if i > 0:    
                    cv2.line(bev, gt_prev_pt, gt_pt, (0,255,0), 1) 
                gt_prev_pt = gt_pt

            out_video.write(bev.astype(np.uint8))
        out_video.release()

