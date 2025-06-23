from src.utils import train, transformer_train_loss, get_transform, get_loaders, get_transforms
from src.models import IntermediatePatch, SigLIPIntermediate, GLIP, GL_RINE
import torch
import os
import datetime
import json

experiment = {
    # Model based
    "backbone": ('ViT-L/14', 1024), # ('hf-hub:timm/ViT-L-16-SigLIP2-256', 1024), # 
    "nproj": 2,
    "proj_dim": 1024, # 512, # 
    # "model": "global",
    # "n_heads": 4,
    # "att_dim": 256,

    # Training based
    "crop": 224, # 256, #
    "imgsize": 224, # 256, #
    "training_set": "progan",
    "contrastive": True,
    "batch_size": 64,
    "classes": os.listdir(f"data/train"), #  ["horse", 'chair', 'car', 'cat'], # 
    "ds_frac": 0.2,
    "lr": 1e-4,
    "lr_step": 2,
    "lr_gamma": 0.5,
    "epochs": 5,
    "factor": 0.2,
    "window_slide": False,
    "save_path": "GLIP/2_nproj_1024_proj_dim",
    # "save_path": "RINE/2_nproj_1024_proj_dim",
    # "save_path": "IntermediatePatch_4class/2_nproj_1024_proj_dim",
}

# with open(f"ckpt/IntermediatePatch/2_nproj_1024_proj_dim/experiment.json", "r") as f:
#     experiment = json.load(f)

# experiment["save_path"] = "IntermediatePatch_seed_42/2_nproj_1024_proj_dim"
# experiment["window_slide"] = False

model = GLIP(
    backbone=experiment["backbone"],
    nproj=experiment["nproj"],
    proj_dim=experiment["proj_dim"],
    device=torch.device("cuda:0"),
)

# model = GL_RINE(
#     backbone=experiment["backbone"],
#     nproj=experiment["nproj"],
#     proj_dim=experiment["proj_dim"],
#     device=torch.device("cuda:0"),
# )

# model = IntermediatePatch(
#     backbone=experiment["backbone"],
#     nproj=experiment["nproj"],
#     proj_dim=experiment["proj_dim"],
#     device=torch.device("cuda:0"),
# )

# model = SigLIPIntermediate(
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
    # seed=42,
    # method="mean"
)
