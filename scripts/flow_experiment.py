from src.utils import train
from src.models import FlowModel
import torch
import os

experiment = {
    # Model based
    "backbone": ('ViT-L/14', 1024),
    "flow": "glow",
    "n_steps": 6,
    "n_proj": 2,
    "proj_dim": 512,

    # Training based
    "training_set": "progan",
    "batch_size": 32,
    "classes": os.listdir(f"results/transform_features/train"), # ["horse"], # 
    "ds_frac": 0.5,
    "lr": 1e-4,
    "lr_step": 5,
    "lr_gamma": 0.5,
    "epochs": 2,
    "log_path": "FlowModel/Intermediate_10step_all_classes",
    "save_path": "FlowModel/10step_all_classes",
}
model = FlowModel(
    backbone=experiment["backbone"],
    flow="glow",
    n_steps=experiment["n_steps"],
    n_proj=experiment["n_proj"],
    proj_dim=experiment["proj_dim"],
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
)

def loss_fn(log_probs, labels=None):
    return - log_probs.mean(axis=0)

def score_fn(log_probs):
    return 1 - torch.exp(log_probs)

train(
    experiment=experiment,
    model=model,
    loss_fn=loss_fn,
    epochs=experiment["epochs"],
    workers=12,
    device=torch.device("cuda:0"),
    store=True,
)