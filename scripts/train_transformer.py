from src.utils import train, transformer_train_loss
from src.models import IntermediatePatch, WindowedIntermediatePatch, AttentionIntermediatePatch
import torch
import os
import datetime

experiment = {
    # Model based
    # "backbone": ('ViT-L/14', 1024),
    # "nproj": 3,
    # "proj_dim": 512,
    "n_heads": 4,
    "att_dim": 256,

    # Training based
    "crop": 224,
    "imgsize": 256,
    "training_set": "ldm",
    "contrastive": False,
    "batch_size": 64,
    # "classes": os.listdir(f"data/train"), # ["horse"], # 
    # "ds_frac": 0.2,
    "lr": 1e-4,
    "lr_step": 5,
    "lr_gamma": 0.5,
    "epochs": 2,
    "factor": 0.3,
    "save_path": "WindowedIntermediatePatchLDM_SupConLoss/4_heads_256_att_dim",
    # "save_path": "IntermediatePatchLDM_SupConLoss/3_nproj_512_proj_dim",
}
model = AttentionIntermediatePatch(
    n_heads=experiment["n_heads"],
    att_dim=experiment["att_dim"],
    device=torch.device("cuda:0"),
)
# model = IntermediatePatch(
#     backbone=experiment["backbone"],
#     nproj=experiment["nproj"],
#     proj_dim=experiment["proj_dim"],
#     device=torch.device("cuda:0"),
# )
print(datetime.datetime.now())
train(
    experiment=experiment,
    model=model,
    loss_fn=transformer_train_loss(experiment["factor"], experiment["contrastive"], unsqueeze=False),
    epochs=experiment["epochs"],
    workers=2,
    device=torch.device("cuda:0"),
    store=True,
    # method="mean"
)