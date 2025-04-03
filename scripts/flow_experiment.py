from src.utils import train_flow_experiment
import torch
import os

MAIN_DIR = "/home/babicdom/SID/results/features/"
experiment = {
    "flow": "glow",
    "num_steps": 2,
    "training_set": "progan",
    "batch_size": 32,
    "classes": os.listdir(f"{MAIN_DIR}/train"), # ["horse"], # 
    "lr": 1e-3,
    "epochss": [1],
    "epochs_reduce_lr": [6, 11],
    "savpath": "results/flow",
}

train_flow_experiment(
    experiment=experiment,
    epochss=experiment["epochss"],
    epochs_reduce_lr=experiment["epochs_reduce_lr"],
    workers=12,
    device=torch.device("cuda:0"),
    store=False,
)