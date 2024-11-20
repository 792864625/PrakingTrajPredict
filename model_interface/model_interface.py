from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, TQDMProgressBar
from model_interface.trainer_real import ParkingTrainingModuleReal

def get_parking_model(data_mode, run_mode):
    return ParkingTrainingModuleReal


def setup_callbacks(cfg):
    ckpt_callback = ModelCheckpoint(dirpath=cfg.checkpoint_dir,
                                    monitor='val_loss',
                                    save_top_k=3,
                                    mode='min',
                                    filename='{epoch:02d}-{val_loss:.2f}',
                                    save_last=True)
    progress_bar = TQDMProgressBar()
    model_summary = ModelSummary(max_depth=2)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    return [ckpt_callback, progress_bar, model_summary, lr_monitor]