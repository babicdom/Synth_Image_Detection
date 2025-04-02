from src.utils import extract_clip_features, get_transform, get_loader
import os
import torch
import clip

MAIN_DIR = "/mnt/personal/babicdom"

splits = ["train", "val"]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device=device)
workers = 12
ds_frac = 1

for split in splits:
    all_classes = os.listdir(f"{MAIN_DIR}/data/{split}/")
    print(f"Classes in {split}: {all_classes}")
    for cls in all_classes:
        experiment = {
                "training_set": "progan",
                "classes": [cls],
                "batch_size": 32,
                "featpath": "results/features",
        }
        print(f"\tExtracting features for {experiment['classes']}")
        
        for target in ["real", "fake"]:
            extract_clip_features(
                experiment=experiment,
                model=model,
                preprocess=preprocess,
                split=split,
                ds_frac=ds_frac,
                device=device,
                target=target,
                save=True
            )

for target in ["real", "fake"]:
    experiment = {
                "training_set": "progan",
                "batch_size": 32,
                "featpath": "results/features",
        }
    test_loader = get_loader(
            experiment=experiment,
            split="test",
            transforms=preprocess,
            workers=workers,
            ds_frac=ds_frac,
            target=target
    )
    for g, dl in test_loader:
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