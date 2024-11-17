import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torchvision.transforms import v2
import torch
import wandb
import yaml
from emir.estimators.knife import KNIFE
from emir.estimators.knife_estimator import KNIFEArgs
from torch.utils.data import DataLoader
from utils.data_multifiles import MultiTeacherAlignedEmbeddingDataset
from utils.teachers_dict import teachers_dict, EmbedderFromTorchvision, embedder_size, teachers_dict_vit, EmbedderFromViT, ModelImageTransform
from utils.trainer_gm import *

def get_parser():
    parser = argparse.ArgumentParser()


    # training parameters
    parser.add_argument("--student", type=str, default="PVTv2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--valid-prop", type=float, default=0.1)
    parser.add_argument("--teachers", type=str, default="./EmbeddingsVit")
    parser.add_argument("--num_teachers", type=int, default=11)

    parser.add_argument("--knifes-config", type=str, default="knifes.yaml")

    # other parameters
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--save-name", type=str, default="tmp")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results")

    return parser

def main(args, list_teachers):
    '''Define The Student Model'''
    #checkpoint = torch.load(best_model)
    #model.load_state_dict(checkpoint)

    if args.student in teachers_dict_vit.keys():
        model = EmbedderFromViT(args.student).to(args.device)
        transform = ModelImageTransform(args.student)
    else:
        model = EmbedderFromTorchvision(args.student).to(args.device)
        #model = teachers_dict[args.student].to(args.device)
        transform = v2.Compose([v2.Resize((224,224)),v2.ToTensor(),])

    '''Read The Data:'''
    dataset = MultiTeacherAlignedEmbeddingDataset(teachers_path = args.teachers, list_teachers = list_teachers, transform = transform)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=30,
        drop_last=True,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=30,
    )
    print("done")

    '''Create Teacher KNIFE'''
    
    if os.path.exists(args.knifes_config):
        with open(args.knifes_config, "r") as f:
            knifes_config = yaml.safe_load(f)
            knifes_config = KNIFEArgs(**knifes_config)
    else:
        knifes_config = KNIFEArgs(device=args.device)
        os.makedirs(os.path.dirname(args.knifes_config), exist_ok=True)
        with open(args.knifes_config, "w") as f:
            yaml.dump(knifes_config.__dict__, f)


    knifes = []
    embs_dim = [embedder_size[t] for t in list_teachers]
    for emb_dm in embs_dim:
        knife = KNIFE(
            args=knifes_config,
            zc_dim=embedder_size[args.student],
            zd_dim=emb_dm,
        ).kernel_cond

        knifes.append(knife)

    knifes = torch.nn.ModuleList(knifes)
    print("knives out ;D")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.L1Loss()
    scheduler = None  # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    # optimizer, T_0=(args.num_epochs * 4) // 10, eta_min=args.lr / 100, T_mult=1
    # )


    trainer = TrainerGM(
        model,
        knifes,
        optimizer,
        criterion,
        scheduler=scheduler,
        device=args.device,
        batch_size=args.batch_size,
        wandb=args.wandb,
        embedder_name_list=list(teachers_dict_vit.keys()),
        out_dir=args.out_dir,
    )


    trainer.train(
        train_loader,
        valid_loader,
        args.num_epochs,
        args.log_interval,
    )




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.wandb:
        wandb.init(
            project="distill-vit-" + args.student,
            allow_val_change=True,
        )

        if not wandb.run.name is None:
            args.out_dir = os.path.join(args.out_dir, wandb.run.name)
        print(args.out_dir)
        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("eval_loss", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        for embedder in list(teachers_dict_vit.keys()):
            wandb.define_metric(f"train_loss_{embedder}", step_metric="epoch")
            wandb.define_metric(f"test_loss_{embedder}", step_metric="epoch")
        wandb.config.update(args)

    os.makedirs(args.out_dir, exist_ok=True)
    main(args, list(teachers_dict_vit.keys()))