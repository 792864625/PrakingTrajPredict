import torch
import math
from torch import nn
from timm.models.layers import trunc_normal_
from utils.config import Configuration
    
class TrajectoryDecoderAvgFc(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, cfg.autoregressive_points*2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, encoder_out, tgt):
        batch_size, channels, seq_len = encoder_out.shape
        encoder_out =  encoder_out.view(batch_size, channels, 16, 16)
        encoder_out = self.adaptive_pool(encoder_out)
        encoder_out = torch.flatten(encoder_out, 1)
        pred_normal = self.fc(encoder_out)
        return pred_normal
    
    def predict(self, encoder_out):
        batch_size, channels, seq_len = encoder_out.shape
        encoder_out =  encoder_out.view(batch_size, channels, 16, 16)
        encoder_out = self.adaptive_pool(encoder_out)
        encoder_out = torch.flatten(encoder_out, 1)
        pred_normal = self.fc(encoder_out)
        return pred_normal


class TrajectoryDecoderDETR(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        #Query
        #tgt的content初始化
        self.tgt_content_embeds = nn.Embedding(1, self.cfg.tf_de_dim)
        #tgt 的位置编码
        self.tgt_pos_embeds = nn.Embedding(1, self.cfg.tf_de_dim)

        #Key Value 
        #bev feature的位置编码
        bev_pos_embedding = PositionEmbeddingSine(self.cfg.tf_de_dim//2)
        self.bev_pos_embeds = bev_pos_embedding(torch.zeros(1, self.cfg.tf_de_dim, 16, 16))      #[c, h, w]
        self.bev_pos_embeds = self.bev_pos_embeds.view(self.bev_pos_embeds.shape[0], -1).permute(1,0).unsqueeze(0)                #[s, c]

        #定义transformer decoder结构
        attn_module_layer = nn.TransformerDecoderLayer(self.cfg.tf_de_dim, 8, dim_feedforward=self.cfg.tf_de_dim*2, dropout=0.05, batch_first=True)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, self.cfg.query_en_layers)


        #reg head
        self.reg_branch = nn.Sequential(
            nn.Linear(self.cfg.tf_de_dim, self.cfg.tf_de_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.tf_de_dim, self.cfg.autoregressive_points * 2),
        )

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'tgt_content_embeds' in name or 'tgt_pos_embeds' in name or 'bev_pos_embeds' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, encoder_out):
        batch_size = encoder_out.shape[0]
        cur_device = encoder_out.device
        encoder_out = encoder_out.permute(0,2,1)     #[b,c,s]->[b,s,c]
        bev_feature = encoder_out + self.bev_pos_embeds.to(cur_device)
        
        # index=torch.tensor(0, device=cur_device)
        # tgt_position = self.tgt_pos_embeds(index).view(1,1,self.cfg.tf_de_dim)
        tgt_position = self.tgt_content_embeds.weight[:, None, :] 
        tgt_content = torch.zeros_like(tgt_position).expand(batch_size,1,self.cfg.tf_de_dim)    #DETR
        
        
        tgt_query = tgt_content + tgt_position

        feature = self.attn_module(tgt_query, bev_feature)

        res = self.reg_branch(feature)
        return res.squeeze(dim=1)
    
    def predict(self, encoder_out):
        return self.forward(encoder_out)

class PositionEmbeddingSine(torch.nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("If scale is passed, normalize should be True")
        self.scale = scale or 2 * math.pi

    def forward(self, x):
        # 获取特征图的高宽
        batch_size, _, height, width = x.size()
        # y和x方向上的位置坐标
        y_embed = torch.arange(height, dtype=torch.float32).unsqueeze(1).repeat(1, width)
        x_embed = torch.arange(width, dtype=torch.float32).unsqueeze(0).repeat(height, 1)

        if self.normalize:
            y_embed = y_embed / (height - 1) * self.scale
            x_embed = x_embed / (width - 1) * self.scale

        # 生成正弦和余弦编码
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        return pos
