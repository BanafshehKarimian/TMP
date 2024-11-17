import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import torchvision.models as models
import lightning as pl
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader
from Data.dataset_dict import datasets_dict
from utils.teachers_dict import teachers_dict
from lightning.pytorch.loggers import CSVLogger

from utils.DistributedFriendlyKernel import GaussianCondKernel
from torch.utils.data import DataLoader
from utils.data_multifiles import MultiTeacherAlignedEmbeddingDataset
from utils.pl_model import (
    DistilledEmbedderPLModelAlignedInputs,
    DistilledEmbedderPLModel
)
# os make WANDB_MODE offline
os.environ["WANDB_MODE"] = "offline"
wandb.init(project="distill")



def parse_arguments():
    parser = argparse.ArgumentParser() #model_name, teachers

    parser.add_argument("--embedding_dimension", type=int, default=1000)
    parser.add_argument("--teachers", type=str, default="./Embeddings")
    parser.add_argument("--num_teachers", type=int, default=11)

    # epoch
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)#increase
    # gradient accumulation
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--train_normalized", action="store_true", default=False)

    # lr
    parser.add_argument("--lr", type=float, default=1e-4)

    return parser.parse_args()




@dataclass(frozen=True)
class KernelArg:
    average: str = "var"
    cov_diagonal: str = "var"
    cov_off_diagonal: str = ""

    optimize_mu: bool = False
    cond_modes: int = 1
    use_tanh: bool = True
    init_std: float = 0.01
    ff_residual_connection: bool = False
    ff_activation: str = "relu"
    ff_layer_norm: bool = True
    ff_layers: int = 2
    ff_dim_hidden: int = 0

def main():
    args = parse_arguments()
    kernel_arg = KernelArg()
    model = models.resnet18(pretrained=True)
    dataset = MultiTeacherAlignedEmbeddingDataset(teachers_path=args.teachers)
    logging.log(logging.INFO, f"Dataset: {dataset}")
    teachers_kernels = [
        GaussianCondKernel(
            kernel_arg, zc_dim=args.embedding_dimension, zd_dim=args.embedding_dimension
        )
        for ds in datasets_dict.keys()
    ]

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=6,
        timeout=10000,
        prefetch_factor=512,
        #worker_init_fn=worker_init_fn,#?
    )
    plModel = DistilledEmbedderPLModel(
        model=model,
        teachers_kernels=teachers_kernels,
        lr=args.lr,
        train_normalized=args.train_normalized,
    )
    wandb_logger = WandbLogger(log_model=False)

    # wandb_logger.experiment.config.update(vars(args))

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{wandb.run.id}",
        monitor="train_nll",
        every_n_train_steps=5000,
        mode="min",
        save_top_k=-1,
        save_last=True,
        auto_insert_metric_name=True,
    )
    logging.log(logging.INFO, f"Make the trainer")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cuda",
        devices=1,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=1,
        enable_progress_bar=True,
        logger=[wandb_logger, CSVLogger("logs", name="test")],
        plugins=[LightningEnvironment()],
        precision=16,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        # strategy=DDPStrategy(find_unused_parameters=True),
        # strategy=DDPStrategy(find_unused_parameters=False),
    )

    logging.log(logging.INFO, f"Start training")
    trainer.fit(plModel, train_dataloaders=dataloader, ckpt_path="last")





if __name__ == "__main__":
    main()