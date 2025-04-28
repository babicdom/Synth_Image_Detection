from src.utils import train, transformer_train_loss, bce, supcon
from src.models import Model
import torch
import os

experiment = {
    # Model based
    "backbone": ('ViT-L/14', 1024),
    "n_layers": 4,
    "n_heads": 8,
    "mlp_dim": 1024,
    "att_dim": 512,

    # Training based
    "training_set": "progan",
    "contrastive": False,
    "batch_size": 64,
    "classes": os.listdir(f"results/transform_features/train"), # ["horse"], # 
    "ds_frac": 0.05,
    "lr": 1e-4,
    "lr_step": 5,
    "lr_gamma": 0.5,
    "epochs": 1,
    "factor": 0.2,
    "log_path": "PerPatchModel/4layers_8heads_all_classes",
    "save_path": "PerPatchModel/4layers_8heads_all_classes",
}
model = Model(
    backbone=experiment["backbone"],
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    n_layers=experiment["n_layers"],
    n_heads=experiment["n_heads"],
    mlp_dim=experiment["mlp_dim"],
    att_dim=experiment["att_dim"],
)

train(
    experiment=experiment,
    model=model,
    loss_fn=transformer_train_loss(experiment["factor"], experiment["contrastive"], unsqueeze=True),
    score_fn=lambda x:torch.sigmoid(x[0]).squeeze(),
    epochs=experiment["epochs"],
    workers=12,
    device=torch.device("cuda:0"),
    store=True,
)