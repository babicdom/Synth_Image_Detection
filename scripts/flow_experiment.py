from src.utils import train_flow_experiment
import torch
import os

MAIN_DIR = "/home/babicdom/Synth_Image_Detection/results/transform_features"

experiment = {
    "flow": "glow",
    "num_steps": 10,
    "training_set": "progan",
    "batch_size": 32,
    "classes": os.listdir(f"{MAIN_DIR}/train"), # ["horse"], # 
    "lr": 1e-4,
    "lr_step": 5,
    "lr_gamma": 0.5,
    "epochs": 15,
    "epochs_reduce_lr": [6, 11],
    "savpath": "results/flow",
}

train_flow_experiment(
    experiment=experiment,
    epochs=experiment["epochs"],
    workers=12,
    device=torch.device("cuda:0"),
    store=True,
)