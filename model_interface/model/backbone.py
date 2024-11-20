from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
from torch import nn
from utils.bev_utils import DeepLabHead, UpsamplingConcat, VoxelsSumming, calculate_birds_eye_view_parameters
from utils.config import Configuration
class efficient_net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.C = cfg.bev_encoder_in_channel
        self.downsample = cfg.bev_down_sample
        self.version = cfg.backbone.split('-')[1]

        self.backbone = EfficientNet.from_pretrained(cfg.backbone)
        self.delete_unused_layers()
        if self.version == 'b4':
            self.reduction_channel = [0, 24, 32, 56, 160, 448]
        elif self.version == 'b0':
            self.reduction_channel = [0, 16, 24, 40, 112, 320]
        else:
            raise NotImplementedError
        self.upsampling_out_channel = [0, 48, 64, 128, 512]

        index = np.log2(self.downsample).astype(np.int32)

        self.feature_layer_1 = DeepLabHead(self.reduction_channel[index + 1],
                                           self.reduction_channel[index + 1],
                                           hidden_channel=64)
        self.feature_layer_2 = UpsamplingConcat(self.reduction_channel[index + 1] + self.reduction_channel[index],
                                                256)    #out_channel=256 为了与target feature对齐
        
        self.head_CBR = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        index = np.log2(self.downsample).astype(np.int32)
        input_1 = endpoints['reduction_{}'.format(index + 1)]
        input_2 = endpoints['reduction_{}'.format(index)]

        feature = self.feature_layer_1(input_1)
        feature = self.feature_layer_2(feature, input_2)
        feature = self.head_CBR(feature)
        return feature

    def forward(self, x):
        feature = self.get_features(x)
        
        return feature

