from src.utils import train, get_loaders, get_transforms
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
    "epochs": 4,
    "log_path": "FlowModel/Intermediate_6step_all_classes",
    "save_path": "FlowModel/6step_all_classes",
}
model = FlowModel(
    backbone=experiment["backbone"],
    flow="glow",
    n_steps=experiment["n_steps"],
    n_proj=experiment["n_proj"],
    proj_dim=experiment["proj_dim"],
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
)
tr_train, tr_val, tr_test = get_transforms()
data = get_loaders(
    experiment=experiment,
    transforms_train=tr_train,
    transforms_test=tr_test,
    transforms_val=tr_val,
    target="real",
    workers=12,
)

def loss_fn(log_probs, labels):
    return - log_probs.mean(axis=0)

def score_fn(log_probs):
    return 1 - torch.exp(log_probs)

def nan_hook(module, input, output):
    if isinstance(output, tuple):
        output = output[0]
    if torch.isnan(output).any():
        print(f"NaN in {module.__class__.__name__}")
        raise ValueError

for layer in model.flow.modules():
    layer.register_forward_hook(nan_hook)

train(
    experiment=experiment,
    model=model,
    data=data,
    loss_fn=loss_fn,
    epochs=experiment["epochs"],
    workers=12,
    device=torch.device("cuda:0"),
    store=True,
)