import os
import torch
import random
import pytorch_lightning
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra

_clearm = True
try:
    from clearml import Task
    from clearml import Logger
except:
    _clearm = False

from src.sequential_dataset import SequentialTemplateDataset
from src.model import TemplateModel


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg):
    if cfg.log.clearml:
        assert _clearm, "Please install clearml"
        seed = cfg.seed if cfg.seed >= 0 else random.randint(0, 2**32 - 1)
        cfg.seed = seed
        print("Seed:", seed)
        
        Task.add_requirements("./requirements.txt")
        Task.set_random_seed(seed)
        
        task = Task.init(
            project_name=cfg.clearml.project,
            task_name=cfg.clearml.task,
            tags=cfg.clearml.tags,
            output_uri=cfg.clearml.output_uri,
        )
        Logger.current_logger().set_default_upload_destination(cfg.clearml.media_uri)

        queue = os.environ.get("CLEARML_QUEUE", False) 
        if queue:
            task.close()
            task.reset()
            Task.enqueue(task, queue_name=queue)
            print(f"Task enqueued on {queue}")
            exit()

    if cfg.train.seed < 0:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    
    seed_everything(cfg.train.seed)
    callbacks = list()

    if cfg.train.checkpoints:
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
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.gpus,
        max_epochs=cfg.train.epochs,
        enable_checkpointing=cfg.train.enable_checkpoints
    )
    
    trainer.fit(model, train_dataloader, validation_dataloader)

    if cfg.train.save != "":
        trainer.save_checkpoint(cfg.train.save)

    if cfg.log.clearml and cfg.clearml.save:
        torch.save(trainer.model, "/tmp/model.pth")

if __name__ == "__main__":
    train()
