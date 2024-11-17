import argparse
import torch
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as L
from torchmetrics.functional import auroc
from utils.ModelTesting import TunnerModel
from Data.DataClass import *
import pandas as pd
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
import wandb
import logging
from sklearn.metrics import top_k_accuracy_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from utils.teachers_dict import teachers_dict, EmbedderFromTorchvision, embedder_size, teachers_dict_vit, EmbedderFromViT, ModelImageTransform
from Data.dataset_dict import datasets_dict


def save_predictions(model, output_fname, num_classes):
    prds = torch.cat(model.predictions, dim=0)
    trgs = torch.cat(model.targets, dim=0)

    auc = auroc(prds, trgs, num_classes=num_classes, average='macro', task='multiclass')

    print('AUROC (test)')
    print(auc)

    cols_names = ['class_' + str(i) for i in range(0, num_classes)]

    df = pd.DataFrame(data=prds.cpu().numpy(), columns=cols_names)    
    df['target'] = trgs.cpu().numpy()
    df.to_csv(output_fname, index=False)
    l = []
    for i in range(num_classes):
        l.append(df['class_'+ str(i)])
    preds = np.stack(l).transpose()
    targets = np.array(df['target'])
    print("balanced accuracy, F1 score:")
    print(balanced_accuracy_score(targets, preds.argmax(1)), f1_score(targets, preds.argmax(1), average='micro'), accuracy_score(targets, preds.argmax(1)))

L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "gpu" if torch.cuda.is_available() else "cpu"
num_workers = 8
batch_size = 64
output_base_dir = 'output_teacher'

def finetune(data, model_, output_name, num_classes, num_epoch, dim):
    model = TunnerModel(model=model_, num_classes = num_classes, output_dim = dim)
    wandb.init(project="distill_compare_"+output_name, reinit=True)
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    monitor="val_acc"
    mode='max'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=mode)
    early_stop_callback = EarlyStopping(
            monitor="val_acc",
            min_delta=0.001,
            patience=5,
            verbose=False,
            mode="max",
            
        )
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=mode)
    wandb_logger = WandbLogger(log_model=False)
    trainer = pl.Trainer(
            callbacks=[checkpoint_callback, early_stop_callback],#
            log_every_n_steps=1,
            max_epochs=num_epoch,
            accelerator=device,
            devices=1,
            logger=[wandb_logger, TensorBoardLogger(output_base_dir, name=output_name), CSVLogger("logs_teacher", name=output_name)],#
        )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)
    print(trainer.checkpoint_callback.best_model_path)
    model = TunnerModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes = num_classes, model = model_, output_dim = dim)
    print(trainer.test(model=model, datamodule=data))
    save_predictions(model, os.path.join(output_dir, 'predictions.csv'), num_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--teachers", nargs="+", type=str, default=list(teachers_dict.keys()))
    parser.add_argument("--datasets", nargs="+", type=str, default=list(datasets_dict.keys()))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--student", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")

    args = parser.parse_args()

    for teacher in args.teachers:
        for dataset in args.datasets:
            if teacher in teachers_dict_vit.keys():
                model_ = EmbedderFromViT(teacher)
                transform = ModelImageTransform(teacher)
            else:
                model_ = EmbedderFromTorchvision(teacher)
                transform = None
            if args.student:
                checkpoint = torch.load(args.checkpoint)
                model_.load_state_dict(checkpoint)
            data = data_module_dict[dataset](batch_size, num_workers, transform = transform)
            output_name = teacher + "_" + dataset + "_teacher_finetuning"
            if args.student:
                output_name = teacher + "_" + dataset + "_student_finetuning"
            print(output_name)
            finetune(data, model_, output_name, data_class_dict[dataset], args.epochs, embedder_size[teacher])#