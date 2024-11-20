import torch
from torch import nn

from model_interface.model.bev_encoder import BevEncoder, BevQuery
from model_interface.model.lss_bev_model import LssBevModel
from model_interface.model.backbone import efficient_net
from model_interface.model.backbone_vit import Vit_Backbone, vit_b_16, vit_l_16
from model_interface.model.trajectory_decoder import TrajectoryDecoderAvgFc, TrajectoryDecoderDETR
from utils.config import Configuration


class ParkingModelReal(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        # Camera Encoder
        self.lss_bev_model = LssBevModel(self.cfg)
        self.image_res_encoder = BevEncoder(in_channel=self.cfg.bev_encoder_in_channel)
        self.bev_res_encoder_res18 = BevEncoder(in_channel=3)
        self.bev_res_encoder_efficient = efficient_net(self.cfg)
        self.bev_res_encoder_vit = Vit_Backbone(model = vit_b_16(pretrained = True)) 
        # Target Encoder
        self.target_res_encoder = BevEncoder(in_channel=1)
        # BEV Query
        self.bev_query = BevQuery(self.cfg)
        # Trajectory Decoder
        self.trajectory_decoder = self.get_trajectory_decoder()

    def forward(self, data):
        # Encoder
        bev_feature, pred_depth, bev_target = self.encoder(data, mode="train")
        # Decoder
        pred_traj_point = self.trajectory_decoder(bev_feature)
        return pred_traj_point, pred_depth, bev_target


    def predict_avg_fc(self, data, heatmap=False):
        bev_feature, pred_depth, bev_target = self.encoder(data, mode="predict")

        if heatmap:
            import numpy as np
            import matplotlib.pyplot as plt
            batch_size, channels, seq_len = bev_feature.shape
            feature_map =  bev_feature.view(batch_size, channels, 16, 16)
            feature_map = feature_map[0].cpu().detach().numpy()  # 移动到CPU并转换为NumPy数组

            # 使用平均融合所有通道
            # 这里可以使用不同的融合方法，例如：平均、最大值等
            fused_feature_map = np.mean(feature_map, axis=0)  # 融合后的特征图形状为 [H, W]

            # 归一化到 [0, 1] 范围
            fused_feature_map = (fused_feature_map - fused_feature_map.min()) / (fused_feature_map.max() - fused_feature_map.min())
            plt.imshow(fused_feature_map, cmap='hot', interpolation='nearest')
            plt.savefig('heatmap.png', bbox_inches='tight')
        pred_traj_point = self.trajectory_decoder.predict(bev_feature)
        return pred_traj_point
    def predict_DETR(self, data, heatmap=False):
        bev_feature, pred_depth, bev_target = self.encoder(data, mode="predict")
        pred_traj_point = self.trajectory_decoder.predict(bev_feature)
        return pred_traj_point

    def encoder(self, data, mode):
        if self.cfg.bev_input:  #input is bev
            bev = data['bev'].to(self.cfg.device, non_blocking=True)
            if self.cfg.bev_input_backbone == "resnet":
                bev_camera_encoder = self.bev_res_encoder_res18(bev, flatten=False)
                pred_depth = None
            elif self.cfg.bev_input_backbone == "efficient_net":
                bev_camera_encoder = self.bev_res_encoder_efficient(bev)
                pred_depth = None
            elif self.cfg.bev_input_backbone == "vit":
                bev_camera_encoder = self.bev_res_encoder_vit(bev)
                pred_depth = None    
        else:                   #inputs are camera images
            # Camera Encoder
            images = data['image'].to(self.cfg.device, non_blocking=True)
            intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
            extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
            bev_camera, pred_depth = self.lss_bev_model(images, intrinsics, extrinsics)
            bev_camera_encoder = self.image_res_encoder(bev_camera, flatten=False)
    
        # Target Encoder
        target_point = data['fuzzy_target_point'] if self.cfg.use_fuzzy_target else data['target_point']
        target_point = target_point.to(self.cfg.device, non_blocking=True)
        bev_target = self.get_target_bev(target_point, mode=mode)
        bev_target_encoder = self.target_res_encoder(bev_target, flatten=False)
        
        # Feature Fusion
        bev_feature = self.get_feature_fusion(bev_target_encoder, bev_camera_encoder)   # (b, c, h, w)

        bev_feature = torch.flatten(bev_feature, 2) # (b, dim, seq)

        return bev_feature, pred_depth, bev_target

    def get_target_bev(self, target_point, mode):
        h, w = int((self.cfg.bev_y_bound[1] - self.cfg.bev_y_bound[0]) / self.cfg.bev_y_bound[2]), int((self.cfg.bev_x_bound[1] - self.cfg.bev_x_bound[0]) / self.cfg.bev_x_bound[2])
        b = self.cfg.batch_size if mode == "train" else 1

        # Get target point
        bev_target = torch.zeros((b, 1, h, w), dtype=torch.float).to(self.cfg.device, non_blocking=True)
        x_pixel = (h / 2 + target_point[:, 0] / self.cfg.bev_x_bound[2]).unsqueeze(0).T.int()       # DONE : xy中心反了吗？正方形没有影响   
        y_pixel = (w / 2 + target_point[:, 1] / self.cfg.bev_y_bound[2]).unsqueeze(0).T.int()               #ego坐标系下的target坐标转换成bevfeature上的grid坐标
        target_point = torch.cat([x_pixel, y_pixel], dim=1)

        # Add noise
        if self.cfg.add_noise_to_target and mode == "train":
            noise_threshold = int(self.cfg.target_noise_threshold / self.cfg.bev_x_bound[2])
            noise = (torch.rand_like(target_point, dtype=torch.float) * noise_threshold * 2 - noise_threshold).int()
            target_point += noise

        # Get target point tensor in the BEV view
        for batch in range(b):
            bev_target_batch = bev_target[batch][0]         #切片浅拷贝
            target_point_batch = target_point[batch]
            range_minmax = int(self.cfg.target_range / self.cfg.bev_x_bound[2])         #在target的中心取一个范围， 然后取1
            bev_target_batch[target_point_batch[0] - range_minmax: target_point_batch[0] + range_minmax + 1,    # DONE ：反了？ROS坐标系问题
                             target_point_batch[1] - range_minmax: target_point_batch[1] + range_minmax + 1] = 1.0
        return bev_target
    

    def get_feature_fusion(self, bev_target_encoder, bev_camera_encoder):
        if self.cfg.fusion_method == "query":
            bev_feature = self.bev_query(bev_target_encoder, bev_camera_encoder)
        elif self.cfg.fusion_method == "plus":
            bev_feature = bev_target_encoder + bev_camera_encoder
        elif self.cfg.fusion_method == "concat":
            concat_feature = torch.concatenate([bev_target_encoder, bev_camera_encoder], dim=1)
            conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False).cuda()
            bev_feature = conv(concat_feature)
        else:
            raise ValueError(f"Don't support fusion_method '{self.cfg.fusion_method}'!")
        
        return bev_feature
    
    def get_trajectory_decoder(self):
        if self.cfg.decoder_method =="avg_fc":
            trajectory_decoder = TrajectoryDecoderAvgFc(self.cfg)
        elif self.cfg.decoder_method == "DETR":
            trajectory_decoder = TrajectoryDecoderDETR(self.cfg)
        else:
            raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")
        
        return trajectory_decoder