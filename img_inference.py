import argparse, os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import glob

from utils.config import get_inference_config_obj
from utils.fev2avm import FEV2AVM
from model_interface.inference_raw import ParkingInferenceRaw


# os.environ["CUDA_VISIBLE_DEVICES"]="1"
def draw_result(avm:FEV2AVM, inference_cfg, img_path_dict:dict, pred_traj, target_point, gt_traj):
    bev_x_bound = inference_cfg.train_meta_config.bev_x_bound
    bev_y_bound = inference_cfg.train_meta_config.bev_y_bound
    bev_h, bev_w = int((bev_y_bound[1] - bev_y_bound[0]) / bev_y_bound[2]), int((bev_x_bound[1] - bev_x_bound[0]) / bev_x_bound[2])
    def get_img_pt(pt, out_shape):
        tgt_y = -pt[0] / bev_x_bound[2] + bev_h/2
        tgt_x = -pt[1] / bev_y_bound[2] + bev_w/2
        tgt_x = int(tgt_x / bev_w * out_shape[1])
        tgt_y = int(tgt_y / bev_h * out_shape[0])
        return(tgt_x, tgt_y)
    
    def draw_trajectory(img, traj_pts, color):
        prev_pt = None
        for pt_idx, traj_pt in enumerate(traj_pts):
            traj_pt = get_img_pt(traj_pt, out.shape)
            cv2.circle(img, traj_pt, 1, (255, 255, 255), 2)
            # if pt_idx > 0:
            #     cv2.line(img, prev_pt, traj_pt, color, 2)
            prev_pt = traj_pt
        return img
    
    # process
    out = avm.apply(img_path_dict)

    # draw target
    tgt_pt = get_img_pt(target_point, out.shape)
    cv2.circle(out, tgt_pt, 10, (0, 255, 0), -1)

    # draw gt_traj
    out = draw_trajectory(out, gt_traj.reshape(-1, 2), (255, 255, 255))

    # draw predict
    # out = draw_trajectory(out, pred_traj.reshape(-1, 2), (255, 0, 0))

    return out

if __name__ == "__main__":
    # import ipdb
    # ipdb.set_trace()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--inference_config_path', default="./config/inference_real.yaml", type=str)
    arg_parser.add_argument('--tasks_path', default="e2e_dataset/train/demo_bag/", type=str)
    arg_parser.add_argument('--out_dir', default="./demo_output/test", type=str)
    args = arg_parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)


    inference_cfg = get_inference_config_obj(args.inference_config_path)

    parking_model = ParkingInferenceRaw(inference_cfg)
    avm = FEV2AVM("./config", 1080, 1080, 2)

    # task test
    images_tag = parking_model.images_tag
    task_paths = [os.path.join(args.tasks_path, task_) for task_ in os.listdir(args.tasks_path)]
    for task_path in task_paths:

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(os.path.join(args.out_dir, task_path.split('/')[-1] + '.mp4'), fourcc, 30, (1080, 1080))
        print(f'frame id : {task_path}')
        frame_num = len(glob.glob(os.path.join(task_path, 'trajectory/*.npy')))
        for frame_idx in tqdm(range(frame_num), desc = os.path.basename(task_path)):
            file_prefix = f"{str(frame_idx).zfill(4)}"

            # inference input
            img_path_dict = {}
            for tag in images_tag:
                img_path_dict[tag] = os.path.join(task_path, tag, file_prefix + '.png')
            traj_file = os.path.join(task_path, 'trajectory', file_prefix + '.npy')
            traj_data = np.load(traj_file, allow_pickle=True).item()
            target_point = traj_data['target']

            # out_traj = parking_model.predict(img_path_dict, target_point, is_raw=True)
            # print(out_traj.shape)
            out_traj = None
            out = draw_result(avm, inference_cfg, img_path_dict, out_traj, target_point, traj_data['trajectory'])

            # save
            # out_file = os.path.join(args.out_dir, file_prefix + '.jpg')
            # cv2.imwrite(out_file, out)

            out_video.write(out)
        out_video.release()