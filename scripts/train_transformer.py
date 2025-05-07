from src.utils import train, transformer_train_loss
from src.models import Model
import torch
import os
import datetime

experiment = {
    # Model based
    "backbone": ('hf-hub:timm/ViT-L-16-SigLIP2-256', 1024),
    "nproj": 3,
    "proj_dim": 512,
    "crop": 256,
    "resize": 256,

    # Training based
    "training_set": "progan",
    "contrastive": False,
    "batch_size": 64,
    "classes": os.listdir(f"data/train"), # ["horse"], # 
    "ds_frac": 0.2,
    "lr": 1e-4,
    "lr_step": 5,
    "lr_gamma": 0.5,
    "epochs": 2,
    "factor": 0.2,
    "save_path": "IntermediatePatchSigLIP/3_nproj_512_proj_dim",
}
model = Model(
    backbone=experiment["backbone"],
    nproj=experiment["nproj"],
    proj_dim=experiment["proj_dim"],
    device=torch.device("cuda:0"),
)
print(datetime.datetime.now())
train(
    experiment=experiment,
    model=model,
    loss_fn=transformer_train_loss(experiment["factor"], experiment["contrastive"], unsqueeze=True),
    epochs=experiment["epochs"],
    workers=12,
    device=torch.device("cuda:0"),
    store=True,
    method="mean"
)