from src.utils import eval_model
from src.models import Model
import torch
import pickle
import os


experiment = pickle.load(
    open(f"ckpt/Model/4layers_8heads_all_classes/experiment.pickle", "rb")
)
model = Model(
    backbone=experiment["backbone"],
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    n_layers=experiment["n_layers"],
    n_heads=experiment["n_heads"],
    mlp_dim=experiment["mlp_dim"],
    att_dim=experiment["att_dim"],
)
model.load_state_dict(
    torch.load(f"ckpt/Model/4layers_8heads_all_classes/train.pth", map_location="cuda:0")
)

eval_model(
    experiment=experiment,
    model=model,
    score_fn=lambda x:torch.sigmoid(x[0]).squeeze(),
)