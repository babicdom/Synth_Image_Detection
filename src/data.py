from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import pandas as pd
import random
from src.perturbation import perturbation

class TrainingDataset(Dataset):
    def __init__(self, split, classes=None, transforms=None, ds_frac=None, target="both"):
        self.real = [
            (f"data/{split}/{y}/0_real/{x}", 0)
            for y in classes
            for x in os.listdir(f"data/{split}/{y}/0_real")
        ]
        self.fake = [
            (f"data/{split}/{y}/1_fake/{x}", 1)
            for y in classes
            for x in os.listdir(f"data/{split}/{y}/1_fake")
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
            (f"data/train/{x.split('_')[0]}/0_real/{x.split('_')[1]}", 0)
            for x in pd.read_csv(
                f"data/latent_diffusion_trainingset/{split}/real_lsun.txt",
                header=None,
            )
            .values.reshape(-1)
            .tolist()
        ] + [
            (
                (
                    f"data/coco/train2014/COCO_train2014_{x}"
                    if os.path.exists(f"data/coco/train2014/COCO_train2014_{x}")
                    else f"data/coco/val2014/COCO_val2014_{x}"
                ),
                0,
            )
            for x in pd.read_csv(
                f"data/latent_diffusion_trainingset/{split}/real_coco.txt", header=None
            )
            .values.reshape(-1)
            .tolist()
        ]
        fake_dir = f"data/latent_diffusion_trainingset/"
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
                (f"data/test/{generator}/{y}/0_real/{x}", 0)
                for y in os.listdir(f"data/test/{generator}")
                for x in os.listdir(f"data/test/{generator}/{y}/0_real")
            ]
            self.fake = [
                (f"data/test/{generator}/{y}/1_fake/{x}", 1)
                for y in os.listdir(f"data/test/{generator}")
                for x in os.listdir(f"data/test/{generator}/{y}/1_fake")
            ]
        elif "diffusion_datasets/guided" in generator:
            self.real = [
                (f"data/test/diffusion_datasets/imagenet/0_real/{x}", 0)
                for x in os.listdir(f"data/test/diffusion_datasets/imagenet/0_real")
            ]
            self.fake = [
                (f"data/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"data/test/{generator}/1_fake")
            ]
        elif (
            "diffusion_datasets/ldm" in generator
            or "diffusion_datasets/glide" in generator
            or "diffusion_datasets/dalle" in generator
        ):
            self.real = [
                (f"data/test/diffusion_datasets/laion/0_real/{x}", 0)
                for x in os.listdir(f"data/test/diffusion_datasets/laion/0_real")
            ]
            self.fake = [
                (f"data/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"data/test/{generator}/1_fake")
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
                    "whichfaceisreal"
                ]
            ]
        ):
            self.real = [
                (f"data/test/{generator}/0_real/{x}", 0)
                for x in os.listdir(f"data/test/{generator}/0_real")
            ]
            self.fake = [
                (f"data/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"data/test/{generator}/1_fake")
            ]
        elif any(
            [
                x in generator
                for x in [
                    "synthbuster/dalle",
                    "synthbuster/stable-diffusion",
                    "synthbuster/glide",
                    "synthbuster/firefly",
                    "synthbuster/midjourney-v5",
                ]
            ]
        ):
            self.real = [(f"data/test/synthbuster/raise/0_real/{x}", 0) for x in os.listdir("data/test/synthbuster/raise/0_real")]
            self.fake = [
                (f"data/test/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"data/test/{generator}/1_fake")
                if all([y not in x for y in [".txt", ".py"]])
            ]
        elif any(
            [
                x in generator
                for x in [
                    "flux",
                    "gigagan",
                    "midjourney-v6.1",
                    "stable-diffusion-3",
                ]
            ]
        ):
            self.real = [
                (f"data/test/diffusion_datasets/laion/0_real/{x}", 0)
                for x in os.listdir(f"data/test/diffusion_datasets/laion/0_real")
            ]
            self.fake = [
                (f"data/test/spai/{generator}/1_fake/{x}", 1)
                for x in os.listdir(f"data/test/spai/{generator}/1_fake")
            ]

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
        return [image, target, image_path]



class TestDataset(Dataset):
    def __init__(self, data_paths, transforms=None, perturb=None, target="both"):
        self.real, self.fake = self.read_paths(data_paths)

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
        return [image, target, image_path]
    
    def read_paths(self, data_paths):
        real_list = []
        fake_list = []
        for path in set(data_paths):
            real_list += self.get_list(path, must_contain='0_real')
            fake_list += self.get_list(path, must_contain='1_fake')
        real_list = [(x, 0) for x in real_list]
        fake_list = [(x, 1) for x in fake_list]
        return real_list, fake_list
    
    def get_list(self, path, must_contain='', exts=["png", "jpg", "JPEG", "jpeg", "bmp", "tif", "webp"]):
        image_list = [] 
        for r, _, f in os.walk(path):
            for file in f:
                if (file.split('.')[1] in exts) and (must_contain in os.path.join(r, file)):
                    image_list.append(os.path.join(r, file))

        return image_list
    
class TrainingDatasetFreq(Dataset):
    def __init__(self, split, classes=None, transforms=None, ds_frac=None):
        self.images = [
            f"data/{split}/{y}/0_real/{x}"
            for y in classes
            for x in os.listdir(f"data/{split}/{y}/0_real")
        ]
        
        random.shuffle(self.images)
        if ds_frac is not None:
            self.images = self.images[: int(len(self.images) * ds_frac)]

        self.target = torch.randint(0, 2, (len(self.images), )).tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.images[idx]
        target = self.target[idx]
        image = Image.open(image_path).convert("RGB")
    
        if self.transforms is not None:
            image = self.transforms[target](image)

        return [image, target]

if __name__ == "__main__":
    # Example usage
    dataset = TestDataset(
        data_paths=["data/test/synthbuster/raise/0_real", "data/test/synthbuster/dalle2/1_fake"],
    )
    print(len(dataset))