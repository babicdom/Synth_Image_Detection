import os
import torch
import tqdm
from src.utils import get_transform, get_loader

device = "cuda:0"
experiment = {
    "training_set": "ldm",
    "batch_size": 64,
}

transform = get_transform("other")
loaders = get_loader(
    experiment=experiment,
    split="train",
    transforms=transform,
    workers=12
)
print(len(loaders.images))
for data in loaders:
    pass
print(loaders.shapes)
# for g, dl in loaders:
#         print(f'Generator {g} - Fake: {len(dl.dataset.fake)}, Real: {len(dl.dataset.real)}, Total: {len(dl.dataset.images)}')
#         for data in dl:
#             pass
        
#         print(dl.dataset.shapes)