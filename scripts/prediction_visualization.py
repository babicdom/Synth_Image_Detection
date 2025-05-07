import os
import torch
import tqdm
from src.utils import get_transform, get_loader, save_worst_predictions, find_best_acc_threshold
import pickle
from src.models import CLIPformer
from src.data import EvaluationDataset
from torch.utils.data import DataLoader

device = "cuda:0"
experiment = {
    "training_set": "progan",
    "batch_size": 64,

    "save_path": "PerPatchModel/4layers_8heads_all_classes",
}

experiment = pickle.load(
    open(f"ckpt/PerPatchModel/4layers_8heads_all_classes/experiment.pickle", "rb")
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
    torch.load(f"ckpt/PerPatchModel/4layers_8heads_all_classes/train.pth", map_location="cuda:0")
)

transform = get_transform("val")
g = "stable-diffusion-3"
loader = DataLoader(
                    EvaluationDataset(g, transforms=transform, target="both"),
                    batch_size=16,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                )

save_worst_predictions(
    experiment=experiment,
    model=model,
    gen_name=g,
    dl=loader,
    device=device,
    threshold=0.5,
)