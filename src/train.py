import os
import argparse
from clearml import Task

from config import Config
from datamodule import DataModule
from lightning_module import BertClassifier

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config, config_file):
    task = Task.init(project_name='BERT', task_name='BERT')

    checkpoint_callback = ModelCheckpoint(
        filename='bert-classifier-{epoch}-{step}-{val_loss:.4f}',
        save_top_k=1,
        verbose=True,
        monitor=config.monitor_metric,
    )

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

    datamodule = DataModule(config.data_config)
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=1,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ],
    )

    task.upload_artifact(
        name='config_file',
        artifact_object=config_file,
    )
    trainer.fit(model=BertClassifier(config), datamodule=datamodule)


if __name__ == '__main__':
    args = arg_parse()
    config = Config.from_yaml(args.config_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.devices
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(config.seed, workers=True)

    train(config, args.config_file)
