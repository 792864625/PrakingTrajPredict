import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger

from dataset_interface.dataloader import ParkingDataloaderModule
from model_interface.model_interface import get_parking_model, setup_callbacks
from utils.config import get_train_config_obj
from utils.decorator_train import finish, init



def decorator_function(train_function):
    def wrapper_function(*args, **kwargs):
        init(*args, **kwargs)
        train_function(*args, **kwargs)
        finish(*args, **kwargs)
    return wrapper_function


@decorator_function
def train(config_obj):
    if config_obj.log == 'tensorboard':
        logger = TensorBoardLogger(save_dir=config_obj.log_dir, default_hp_metric=False)
    else:
        logger =  CSVLogger(save_dir = config_obj.log_dir)

    parking_trainer = Trainer(callbacks=setup_callbacks(config_obj),
                              num_sanity_val_steps=0,           # 禁用了sanity_val 训练开始前不会随机选batch进行模型稳定性测试
                              logger=logger,
                              accelerator='gpu',
                              strategy='ddp' if config_obj.num_gpus > 1 else None,
                              devices=config_obj.num_gpus,
                              max_epochs=config_obj.epochs,
                              log_every_n_steps=config_obj.log_every_n_steps,
                              check_val_every_n_epoch=config_obj.check_val_every_n_epoch,
                              profiler='simple')
    ParkingTrainingModelModule = get_parking_model(data_mode=config_obj.data_mode, run_mode="train")

    model = ParkingTrainingModelModule(config_obj)
    data = ParkingDataloaderModule(config_obj)
    if config_obj.valid_before_training:
        parking_trainer.validate(model=model, dataloaders=data)         #训练前先validation计算指标

    print('start training --------------------------------------------- ')
    parking_trainer.fit(model=model, datamodule=data, ckpt_path=config_obj.resume_path)

def main():
    seed_everything(16)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', default='./config/training_real.yaml', type=str)
    args = arg_parser.parse_args()
    config_path = args.config
    config_obj = get_train_config_obj(config_path)

    print(f"config ----- {config_obj}")

    train(config_obj)


if __name__ == '__main__':
    
    main()