import argparse, os
import numpy as np
import torch
from utils.config import get_inference_config_obj
from utils.camera_utils import CameraInfoParser
from model_interface.model.lss_bev_model import LssBevModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    # import ipdb
    # ipdb.set_trace()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--inference_config_path', default="./config/inference_real.yaml", type=str)
    args = arg_parser.parse_args()
    inference_cfg = get_inference_config_obj(args.inference_config_path)
    images_tag = ("rgb_front", "rgb_left", "rgb_right", "rgb_rear")

    lss_model = LssBevModel(inference_cfg.train_meta_config)
    lss_model.cuda()
    camera_info_obj = CameraInfoParser(task_index=-1, parser_dir=inference_cfg.cam_info_dir)
    intrinsic = camera_info_obj.intrinsic
    extrinsic = camera_info_obj.extrinsic

    # scale intrinsic
    intrinsics = []
    extrinsics = []
    h, w = inference_cfg.train_meta_config.final_dim
    for tag in images_tag:
        cam_w = intrinsic[tag][0, 2]*2
        cam_h = intrinsic[tag][1, 2]*2
        scale_w = w / float(cam_w)
        scale_h = h / float(cam_h)
        intrinsic[tag][0, 0] *= scale_w
        intrinsic[tag][0, 2] *= scale_w
        intrinsic[tag][1, 1] *= scale_h
        intrinsic[tag][1, 2] *= scale_h
        intrinsics.append(torch.from_numpy(intrinsic[tag][np.newaxis].astype(np.float32)))
        extrinsics.append(torch.from_numpy(extrinsic[tag][np.newaxis].astype(np.float32)))
    intrinsics = torch.cat(intrinsics, dim=0).unsqueeze(0)
    extrinsics = torch.cat(extrinsics, dim=0).unsqueeze(0)

    # compute points
    points = lss_model.get_geometry(intrinsics.cuda(), extrinsics.cuda())
    points = points.cpu()

    # draw points
    fig = plt.figure(figsize=(25, 13))
    ax = fig.add_subplot(121)
    for cami, tag in enumerate(images_tag):
        # ax.plot(-points[0, cami, :, :, :, 1].view(-1), points[0, cami, :, :, :, 0].view(-1), '.', label = tag)
        ax.scatter(-points[0, cami, :, :, :, 1].view(-1), points[0, cami, :, :, :, 0].view(-1), 2, label = tag)
    plt.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    # plt.xlim((-50, 50))
    # plt.ylim((-50, 50))

    # 3D
    bx = fig.add_subplot(122, projection='3d')
    for cami, tag in enumerate(images_tag):
        scatter = bx.scatter(-points[0, cami, :, :, :, 1].view(-1), 
                             points[0, cami, :, :, :, 0].view(-1), 
                             points[0, cami, :, :, :, 2].view(-1), s=2, label = tag)
    plt.legend(loc='upper right')
    bx.set_xlabel('X Axis')
    bx.set_ylabel('Y Axis')
    bx.set_zlabel('Z Axis')
    plt.savefig("lss_check.jpg")
    plt.show()