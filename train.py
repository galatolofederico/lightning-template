import os
import torch
import pytorch_lightning
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import hydra

from src.sequential_dataset import SequentialTemplateDataset
from src.model import TemplateModel

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg):
    if cfg.train.seed < 0:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    
    if cfg.log.clearml:
        from clearml import Task
        Task.init(project_name=cfg.clearml.project, task_name=cfg.clearml.task)

    seed_everything(cfg.train.seed)
    
    loggers = list()
    callbacks = list()

    last_checkpoint_callback = ModelCheckpoint(
        filename="last",
        save_last=True,
    )
    min_loss_checkpoint_callback = ModelCheckpoint(
        monitor="train/loss",
        filename="min-loss",
        save_top_k=1,
        mode="min",
    )
    min_val_loss_checkpoint_callback = ModelCheckpoint(
        monitor="validation/loss",
        filename="min-val-loss",
        save_top_k=1,
        mode="min",
    )

    callbacks.extend([
        last_checkpoint_callback,
        min_loss_checkpoint_callback,
        min_val_loss_checkpoint_callback
    ])
    
    train_dataset = SequentialTemplateDataset(cfg, "train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=os.cpu_count()
    )

    validation_dataset = SequentialTemplateDataset(cfg, "validation")
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=os.cpu_count()
    )

    model = TemplateModel(
        **cfg.model,
        lr=cfg.train.lr,
        input_size=cfg.data.dim,
        output_size=2
    )

    trainer = pytorch_lightning.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.gpus,
        max_epochs=cfg.train.epochs,
    )
    
    trainer.fit(model, train_dataloader, validation_dataloader)


if __name__ == "__main__":
    train()
