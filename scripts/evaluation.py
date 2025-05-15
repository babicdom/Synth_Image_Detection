from src.utils import eval_model, get_loader, get_transform
from src.models import CLIPatch, CLIPformer, Model, IntermediatePatch
import torch
import pickle
import os
import datetime

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

# print(datetime.datetime.now())
# eval_model(
#     experiment=experiment,
#     model=model,
#     method="max",
#     device=torch.device("cuda:0"),
# )

# experiment = pickle.load(
#     open(f"ckpt/IntermediatePatchSigLIP/3_nproj_512_proj_dim/experiment.pickle", "rb")
# )
# model = Model(
#     backbone=experiment["backbone"],
#     nproj=experiment["nproj"],
#     proj_dim=experiment["proj_dim"],
#     device=torch.device("cuda:0"),
# )
# model.load_state_dict(
#     torch.load(f"ckpt/IntermediatePatchSigLIP/3_nproj_512_proj_dim/train.pth", map_location="cuda:0")
# )
# tr = get_transform("val", crop=256)
# dl = get_loader(experiment=experiment, split="test", transforms=tr)

experiment = pickle.load(
    open(f"ckpt/PerPatchModel/4layers_8heads_all_classes/experiment.pickle", "rb")
)
experiment["save_path"] = "PerPatchModel/4layers_8heads_all_classes"
model = CLIPatch(
    backbone=experiment["backbone"],
    n_layers=experiment["n_layers"],
    n_heads=experiment["n_heads"],
    mlp_dim=experiment["mlp_dim"],
    device=torch.device("cuda:0"),
)
model.load_state_dict(
    torch.load(f"ckpt/PerPatchModel/4layers_8heads_all_classes/train.pth", map_location="cuda:0")
)

print(datetime.datetime.now())
eval_model(
    experiment=experiment,
    model=model,
    device=torch.device("cuda:0"),
    # method="mean",
    method="max",
)