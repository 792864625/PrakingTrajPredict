from torch import nn
from utils.config import Configuration
import torch
import torch.nn.functional as F
import math, random, time, json

class TokenTrajPointFocalLoss(nn.Module):
    def __init__(self, cfg, alpha=1.0, gamma=2.0, reduction='mean'):
        super(TokenTrajPointFocalLoss, self).__init__()
        self.cfg = cfg
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1
        self.loss_weights = torch.ones(self.cfg.token_nums + self.cfg.append_token)
        self.loss_weights[-2] = 0.2

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing = 0.1, weight=self.loss_weights,ignore_index=self.PAD_token, reduction='none')

    def forward(self, pred, data):
        pred = pred[:, :-1,:]
        pred_traj_point = pred.reshape(-1, pred.shape[-1])
        gt_traj_point_token = data['gt_traj_point_token'][:, 1:-1].reshape(-1).cuda()
        BCE_loss = self.ce_loss(pred_traj_point, gt_traj_point_token)
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

        
class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, data):
        gt = data['gt_traj_point_normal'].cuda()
        loss = F.smooth_l1_loss(pred * 256, gt * 256)
        return loss

class MseLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, pred, data):
        gt = data['gt_traj_point_normal'].cuda()
        loss = F.mse_loss(pred*256, gt*256)
        return loss


class WingLoss(nn.Module):
    def __init__(self, w=10, epsilon=2) -> None:
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * math.log(1 + self.w / self.epsilon)

    def forward(self, pred, data):
        gt = data['gt_traj_point_normal'].cuda()
        delta = (pred - gt).abs() * 256
        delta1 = delta[delta < self.w]
        delta2 = delta[delta >= self.w]
        loss1 = self.w * torch.log(1 + delta1 / self.epsilon)
        loss2 = delta2 - self.C
        loss = (loss1.sum() + loss2.sum()) / (loss1.nelement() +
                                              loss2.nelement())
        return loss



