from src.utils import extract_clip_features, get_transform, get_loader
import os
import torch
import clip

MAIN_DIR = "/mnt/personal/babicdom"
FEAT_PATH = f"results/transform_features"

splits = ["train", "val"]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device=device)
workers = 12
ds_frac = 1
use_transform = True

# for split in splits:
#     all_classes = os.listdir(f"{MAIN_DIR}/data/{split}/")
#     print(f"Classes in {split}: {all_classes}")
#     for cls in all_classes:
#         experiment = {
#                 "training_set": "progan",
#                 "classes": [cls],
#                 "batch_size": 32,
#                 "featpath": FEAT_PATH,
#         }
#         print(f"\tExtracting features for {experiment['classes']}")
        
#         for target in ["real", "fake"]:
#             extract_clip_features(
#                 experiment=experiment,
#                 model=model,
#                 preprocess=preprocess,
#                 split=split,
#                 ds_frac=ds_frac,
#                 device=device,
#                 target=target,
#                 save=True,
#                 use_transform=use_transform,
#             )

for target in ["real"]:# ["real", "fake"]:
    experiment = {
                "training_set": "progan",
                "batch_size": 32,
                "featpath": FEAT_PATH,
        }
    if use_transform:
        transforms = get_transform(
            split="test"
            )
    else:
        transforms = preprocess

    test_loader = get_loader(
            experiment=experiment,
            split="test",
            transforms=transforms,
            workers=workers,
            ds_frac=ds_frac,
            target=target
    )
    for g, dl in test_loader:
        if g not in "diffusion_datasets/guided":
            continue
        experiment['classes'] = [g]
        print(f"\tExtracting features for {g}")

        extract_clip_features(
            experiment=experiment,
            dl=dl,
            model=model,
            preprocess=preprocess,
            split="test",
            ds_frac=ds_frac,
            device=device,
            target=target,
            save=True
        )