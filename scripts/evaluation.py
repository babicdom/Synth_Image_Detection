from src.utils import eval_model
from src.models import CLIPatch, CLIPformer, Model
import torch
import pickle
import os
import datetime

experiment = pickle.load(
    open(f"ckpt/IntermediatePatch/3_nproj_512_proj_dim/experiment.pickle", "rb")
)
model = Model(
    backbone=experiment["backbone"],
    nproj=experiment["nproj"],
    proj_dim=experiment["proj_dim"],
    device=torch.device("cuda:0"),
)
model.load_state_dict(
    torch.load(f"ckpt/IntermediatePatch/3_nproj_512_proj_dim/train.pth", map_location="cuda:0")
)

print(datetime.datetime.now())
eval_model(
    experiment=experiment,
    model=model,
    method="max",
    device=torch.device("cuda:0"),
)