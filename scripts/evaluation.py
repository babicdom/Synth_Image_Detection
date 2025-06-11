from src.utils import eval_model, get_loader, get_transform, image_enlisting_collate_fn
from src.models import IntermediatePatch, SigLIPIntermediate
import torch
import pickle
import os
import datetime
import json

# experiment = pickle.load(
#     open(f"ckpt/IntermediatePatch/3_nproj_512_proj_dim/experiment.pickle", "rb")
# )
# model = IntermediatePatch(
#     backbone=experiment["backbone"],
#     nproj=experiment["nproj"],
#     proj_dim=experiment["proj_dim"],
#     device=torch.device("cuda:0"),
# )
# model.load_state_dict(
#     torch.load(f"ckpt/IntermediatePatch/3_nproj_512_proj_dim/train.pth", map_location="cuda:0")
# )

# ----------------------------------------------------

experiment = json.load(
    open(f"ckpt/IntermediatePatchSigLIP/3_nproj_512_proj_dim/experiment.json", "rb")
)
model = SigLIPIntermediate(
    backbone=experiment["backbone"],
    nproj=experiment["nproj"],
    proj_dim=experiment["proj_dim"],
    device=torch.device("cuda:0"),
)
model.load_state_dict(
    torch.load(f"ckpt/IntermediatePatchSigLIP/3_nproj_512_proj_dim/train.pth", map_location="cuda:0")
)
# experiment["batch_size"] = 8

# transform = get_transform("no_crop_no_norm", imgsize=256)

transform = get_transform("val_siglip")

test = get_loader(
    experiment=experiment,
    split="test",
    transforms=transform,
    # collate_fn=image_enlisting_collate_fn
)

print(datetime.datetime.now())
print(experiment)
eval_model(
    experiment=experiment,
    model=model,
    # test=test,
    device=torch.device("cuda:0"),
    method="mean",
    p=3
    # method="max",
)