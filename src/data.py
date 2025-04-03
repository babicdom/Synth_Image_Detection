from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import pandas as pd
import random
from src.perturbation import perturbation

MAIN_DIR = "/mnt/personal/babicdom"

class TrainingDataset(Dataset):
    def __init__(self, split, classes=None, transforms=None, ds_frac=None, target="both"):
        self.real = [
            (f"{MAIN_DIR}/data/{split}/{y}/0_real/{x}", 0)
            for y in classes
            for x in os.listdir(f"{MAIN_DIR}/data/{split}/{y}/0_real")
        ]
        self.fake = [
            (f"{MAIN_DIR}/data/{split}/{y}/1_fake/{x}", 1)
            for y in classes
            for x in os.listdir(f"{MAIN_DIR}/data/{split}/{y}/1_fake")
        ]

        if target == "both":
            self.images = self.real + self.fake
        elif target == "real":
            self.images = self.real
        elif target == "fake":
            self.images = self.fake
        else:
            raise TypeError('Specify the target data.')
        
        random.shuffle(self.images)
        if ds_frac is not None:
            self.images = self.images[: int(len(self.images) * ds_frac)]

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, target = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return [image, target]


class TrainingDatasetLDM(Dataset):
    def __init__(self, split, transforms=None, target="both"):
        self.real = [
            (f"{MAIN_DIR}/data/train/{x.split('_')[0]}/0_real/{x.split('_')[1]}", 0)
            for x in pd.read_csv(
                f"{MAIN_DIR}/data/latent_diffusion_trainingset/{split}/real_lsun.txt",
                header=None,
            )
            .values.reshape(-1)
            .tolist()
        ] + [
            (
                (
                    f"{MAIN_DIR}/data/coco/train2014/COCO_train2014_{x}"
                    if os.path.exists(f"{MAIN_DIR}/data/coco/train2014/COCO_train2014_{x}")
                    else f"{MAIN_DIR}/data/coco/val2014/COCO_val2014_{x}"
                ),
                0,
            )
            for x in pd.read_csv(
                f"{MAIN_DIR}/data/latent_diffusion_trainingset/{split}/real_coco.txt", header=None
            )
            .values.reshape(-1)
            .tolist()
        ]
        fake_dir = f"{MAIN_DIR}/data/latent_diffusion_trainingset/"
        self.fake = [
            (f"{fake_dir}{split}/{x}/{y}", 1)
            for x in os.listdir(f"{fake_dir}{split}")
            if os.path.isdir(f"{fake_dir}{split}/{x}")
            for y in os.listdir(f"{fake_dir}{split}/{x}")
        ]
        if target == "both":
            self.images = self.real + self.fake
        elif target == "real":
            self.images = self.real
        elif target == "fake":
            self.images = self.fake
        else:
            raise TypeError('Specify the target data.')
        random.shuffle(self.images)

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, target = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return [image, target]


class EvaluationDataset(Dataset):
    def __init__(self, generator, transforms=None, perturb=None, target="both"):
        if generator in ["cyclegan", "progan", "stylegan", "stylegan2"]:
            self.real = [
                (f"{MAIN_DIR}/data/test/{generator}/{y}/0_real/{x}", 0)
                for y in os.listdir(f"{MAIN_DIR}/data/test/{generator}")
                for x in os.listdir(f"{MAIN_DIR}/data/test/{generator}/{y}/0_real")
            ]
            self.fake = [
                (f"{MAIN_DIR}/data/test/{generator}/{y}/1_fake/{x}", 1)
                for y in os.listdir(f"{MAIN_DIR}/data/test/{generator}")
                for x in os.listdir(f"{MAIN_DIR}/data/test/{generator}/{y}/1_fake")
            ]
        elif "diffusion_datasets/guided" in generator:
            self.real = [
                (f"{MAIN_DIR}/data/test/diffusion_datasets/imagenet/0_real/{x}", 0)
                for x in os.listdir(f"{MAIN_DIR}/data/test/diffusion_datasets/imagenet/0_real")
            ]
            self.fake = [
                (f"{MAIN_DIR}/data/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"{MAIN_DIR}/data/test/{generator}/1_fake")
            ]
        elif (
            "diffusion_datasets/ldm" in generator
            or "diffusion_datasets/glide" in generator
            or "diffusion_datasets/dalle" in generator
        ):
            self.real = [
                (f"{MAIN_DIR}/data/test/diffusion_datasets/laion/0_real/{x}", 0)
                for x in os.listdir(f"{MAIN_DIR}/data/test/diffusion_datasets/laion/0_real")
            ]
            self.fake = [
                (f"{MAIN_DIR}/data/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"{MAIN_DIR}/data/test/{generator}/1_fake")
            ]
        elif any(
            [
                x in generator
                for x in [
                    "biggan",
                    "stargan",
                    "gaugan",
                    "deepfake",
                    "seeingdark",
                    "san",
                    "crn",
                    "imle",
                ]
            ]
        ):
            self.real = [
                (f"{MAIN_DIR}/data/test/{generator}/0_real/{x}", 0)
                for x in os.listdir(f"{MAIN_DIR}/data/test/{generator}/0_real")
            ]
            self.fake = [
                (f"{MAIN_DIR}/data/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"{MAIN_DIR}/data/test/{generator}/1_fake")
            ]
        elif any(
            [
                x in generator
                for x in [
                    "dalle2",
                    "dalle3",
                    "stable-diffusion-1-3",
                    "stable-diffusion-1-4",
                    "stable-diffusion-2",
                    "stable-diffusion-xl",
                    "glide",
                    "firefly",
                    "midjourney-v5",
                ]
            ]
        ):
            pass
            # self.real = [(f"{MAIN_DIR}/data/RAISEpng/{x}", 0) for x in os.listdir("data/RAISEpng")]
            # self.fake = [
            #     (f"data/synthbuster/{generator}/{x}", 1)
            #     for x in os.listdir(f"data/synthbuster/{generator}")
            #     if all([y not in x for y in [".txt", ".py"]])
            # ]

        if target == "both":
            self.images = self.real + self.fake
        elif target == "real":
            self.images = self.real
        elif target == "fake":
            self.images = self.fake
        else:
            raise TypeError('Specify the target data.')

        self.transforms = transforms
        self.perturb = perturb

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, target = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None and self.perturb is None:
            image = self.transforms(image)
        elif self.transforms is not None and self.perturb is not None:
            if random.random() < 0.5:
                image = perturbation(self.perturb)(image)
            else:
                image = self.transforms(image)
        return [image, target]


FEAT_DIR = "/home/babicdom/SID/results/features"

class FeatureDataset(Dataset):
    def __init__(self, split, classes=None, ds_frac=None, target="both"):
        self.real = [(x, 0) for x in torch.cat(
            [
                torch.load(f"{FEAT_DIR}/{split}/{y}/real/features.pt")
                for y in classes
            ]
        )]

        self.fake = [(x, 1) for x in torch.cat(
            [
                torch.load(f"{FEAT_DIR}/{split}/{y}/fake/features.pt")
                for y in classes
            ]
        )]

        if target == "both":
            self.features = self.real + self.fake
        elif target == "real":
            self.features = self.real
        elif target == "fake":
            self.features = self.fake
        else:
            raise TypeError('Specify the target data.')
        
        random.shuffle(self.features)
        if ds_frac is not None:
            self.features = self.features[: int(len(self.features) * ds_frac)]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.features[idx]