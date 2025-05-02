from src.utils import eval_model
from src.models import CLIPatch, CLIPformer
import torch
import pickle
import os
import datetime

experiment = pickle.load(
    open(f"ckpt/Model/4layers_8heads_all_classes/experiment.pickle", "rb")
)
model = CLIPformer(
    backbone=experiment["backbone"],
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    n_layers=experiment["n_layers"],
    n_heads=experiment["n_heads"],
    att_dim=experiment["att_dim"],
    mlp_dim=experiment["mlp_dim"],
)
model.load_state_dict(
    torch.load(f"ckpt/Model/4layers_8heads_all_classes/train.pth", map_location="cuda:0")
)

print(datetime.datetime.now())
eval_model(
    experiment=experiment,
    model=model,
    method="max"
)