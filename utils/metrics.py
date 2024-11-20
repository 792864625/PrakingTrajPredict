
import numpy as np
import torch

from utils.config import Configuration
from utils.trajectory_utils import TrajectoryDistance


class CustomizedMetric:
    def __init__(self, cfg: Configuration, pred_traj_point, batch) -> None:
        self.cfg = cfg
        self.distance_dict = self.calculate_distance(pred_traj_point, batch)

    def calculate_distance(self, pred_traj_point, batch):
        distance_dict = {}
        if self.cfg.decoder_method == "avg_fc" or self.cfg.decoder_method == "DETR":
            gt_points = batch['gt_traj_point']
            prediction_points_np = []
            gt_points_np = []
            for index in range(self.cfg.batch_size):
                # [30,2]
                gt_points_np.append(np.array(gt_points[index].view(-1, self.cfg.item_number).cpu()))
                # pred 从normal 转换成 ros 坐标系   
                pred = pred_traj_point[index].view(-1, self.cfg.item_number)
                pred *= 12.0
                pred_ros = torch.empty_like(pred)
                pred_ros[:,0] = 6.0 - pred[:,1]
                pred_ros[:,1] = 6.0 - pred[:,0]
                prediction_points_np.append(np.array(pred_ros.cpu()))
        else:
            raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")

        l2_list, haus_list, fourier_difference = [], [], []
        for index in range(self.cfg.batch_size):
            distance_obj = TrajectoryDistance(prediction_points_np[index], gt_points_np[index])
            if distance_obj.get_len() < 1:
                continue
            l2_list.append(distance_obj.get_l2_distance())
            if distance_obj.get_len() > 1:
                haus_list.append(distance_obj.get_haus_distance())
                fourier_difference.append(distance_obj.get_fourier_difference())
        if len(l2_list) > 0:
            distance_dict.update({"L2_distance": np.mean(l2_list)})
        if len(haus_list) > 0:
            distance_dict.update({"hausdorff_distance": np.mean(haus_list)})
        if len(fourier_difference) > 0:
            distance_dict.update({"fourier_difference": np.mean(fourier_difference)})
        return distance_dict



