from src.utils import extract_clip_features
import os
import torch

MAIN_DIR = "/mnt/personal/babicdom"

splits = ["train", "test", "val"]
targets = []
device = "cuda:0" if torch.cuda.is_available() else "cpu"

for split in splits:
    all_classes = os.listdir(f"{MAIN_DIR}/data/{split}/")

    for cls in all_classes:
        experiment = {
                "training_set": "progan",
                "backbone": ("ViT-L/14", 1024),
                "classes": [cls],
                "batch_size": 32,
                "featpath": "results/features",
        }
        extract_clip_features(
            experiment=experiment,
            split=split,
            ds_frac=1,
            device=device,
            target="real",
            save=True
        )
        extract_clip_features(
            experiment=experiment,
            split=split,
            ds_frac=1,
            device=device,
            target="fake",
            save=True
        )



