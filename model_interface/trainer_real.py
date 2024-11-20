from collections import OrderedDict
import pytorch_lightning as pl
import torch

from loss.traj_point_loss import TokenTrajPointFocalLoss, SmoothL1Loss, WingLoss
from model_interface.model.parking_model_real import ParkingModelReal
from utils.config import Configuration
from utils.metrics import CustomizedMetric


class ParkingTrainingModuleReal(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super(ParkingTrainingModuleReal, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.traj_point_loss_func = self.get_loss_function()
        self.parking_model = ParkingModelReal(self.cfg)
        if cfg.pretrain:
            ckpt = torch.load(cfg.pretrain_model_path, map_location='cuda:0')
            state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])
            self.parking_model.load_state_dict(state_dict)

    def load_model(self, parking_pth_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = ParkingModelReal(self.cfg.train_meta_config)

        ckpt = torch.load(parking_pth_path, map_location='cuda:0')
        state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def training_step(self, batch, batch_idx):
        loss_dict = {}
        pred_traj_point, _, _ = self.parking_model(batch)
        train_loss = self.traj_point_loss_func(pred_traj_point, batch)        
        loss_dict.update({"train_loss": train_loss})
        self.log_dict(loss_dict)
        return train_loss
    
    def on_train_epoch_end(self):
        current_lr = self.optimizers().param_groups[0]['lr']
        print(f'Learning rate at epoch end: {current_lr:.6f}')

    def validation_step(self, batch, batch_idx):
        val_loss_dict = {}
        pred_traj_point, _, _ = self.parking_model(batch)
        val_loss = self.traj_point_loss_func(pred_traj_point, batch)
        val_loss_dict.update({"val_loss": val_loss})
        customized_metric = CustomizedMetric(self.cfg, pred_traj_point, batch)
        val_loss_dict.update(customized_metric.distance_dict)
        # print(f'batch_idx:{batch_idx}    loss_metric:{val_loss_dict}')
        self.log_dict(val_loss_dict)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.cfg.learning_rate,
                                     weight_decay=self.cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.cfg.epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def get_loss_function(self):
        traj_point_loss_func = None
        if self.cfg.decoder_method == "avg_fc":
            traj_point_loss_func = SmoothL1Loss()
            # traj_point_loss_func = WingLoss()
        elif self.cfg.decoder_method == "DETR":
            traj_point_loss_func = SmoothL1Loss()
        else:
            raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")
        return traj_point_loss_func