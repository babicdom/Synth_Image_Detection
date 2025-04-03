import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import clip
import tqdm

import os
from io import BytesIO
import pickle
import copy
import json
import random
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import accuracy_score, average_precision_score

from src.data import TrainingDataset, TrainingDatasetLDM, EvaluationDataset, FeatureDataset
from src.models import Model
from src.ablations import ModelAblations
from src.nf import MiniGlow, NormalizingFlow

import matplotlib.pyplot as plt

def get_transform(split="train"):
    if split == "train":
        return transforms.Compose(
            [
                transforms.Lambda(lambda img: data_augment(img)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    elif split == "val":
        return transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    elif split == "test":
        return transforms.Compose(
            [
                transforms.TenCrop(224),
                transforms.Lambda(
                    lambda crops: torch.stack(
                        [transforms.PILToTensor()(crop) for crop in crops]
                    )
                ),
                transforms.Lambda(lambda x: x / 255),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        raise ValueError("split must be one of train, val, test")
    
def get_transforms():
    transforms_train = get_transform("train")
    transforms_val = get_transform("val")
    transforms_test = get_transform("test")
    return transforms_train, transforms_val, transforms_test

def get_loader(
    experiment, split, transforms, workers, ds_frac=None, target="both"
):
    if experiment["training_set"] == "progan":
        generators = get_generators()
        if split == "train":
            return DataLoader(
                    TrainingDataset(
                        split="train",
                        classes=experiment["classes"],
                        transforms=transforms,
                        ds_frac=ds_frac,
                        target=target
                    ),
                    batch_size=experiment["batch_size"],
                    shuffle=True,
                    num_workers=workers,
                    pin_memory=True,
                    drop_last=False,
                )
        if split == "val":
            return DataLoader(
                    TrainingDataset(
                        split="val", 
                        classes=experiment["classes"], 
                        transforms=transforms, 
                        target=target
                    ),
                    batch_size=experiment["batch_size"],
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=True,
                    drop_last=False,
                )
    elif experiment["training_set"] == "ldm":
        generators = get_generators("synthbuster")
        if split == "train":
            return DataLoader(
                    TrainingDatasetLDM(
                        split="train", 
                        transforms=transforms, 
                        target=target),
                    batch_size=experiment["batch_size"],
                    shuffle=True,
                    num_workers=workers,
                    pin_memory=True,
                    drop_last=False,
                )
        if split == "val":
            return DataLoader(
                    TrainingDatasetLDM(
                        split="valid", 
                        transforms=transforms, 
                        target=target),
                    batch_size=experiment["batch_size"],
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=True,
                    drop_last=False,
                )
    if split == "test":
        return [
            (
                g,
                DataLoader(
                    EvaluationDataset(g, transforms=transforms, target=target),
                    batch_size=(
                        experiment["batch_size"]
                        if experiment["training_set"] == "progan"
                        else 16
                    ),
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=True,
                    drop_last=False,
                ),
            )
            for g in generators
        ]
    else:
        raise ValueError("split must be one of train, val, test")
    
def get_loaders(
    experiment, transforms_train, transforms_val, transforms_test, workers, ds_frac=None, target="both"
):
    train = get_loader(
        experiment,
        split="train",
        transforms=transforms_train,
        workers=workers,
        ds_frac=ds_frac,
        target=target
    )
    val = get_loader(
        experiment,
        split="val",
        transforms=transforms_val,
        workers=workers,
        ds_frac=ds_frac,
        target=target
    )
    test = get_loader(
        experiment,
        split="test",
        transforms=transforms_test,
        workers=workers,
        ds_frac=ds_frac,
        target=target
    )
    return train, val, test

def train_one_experiment(
    experiment,
    epochss,
    epochs_reduce_lr,
    transforms_train,
    transforms_val,
    transforms_test,
    workers,
    device,
    without=None,  # None, contrastive, alpha, intermediate
    store=False,
    ds_frac=None,
):
    seed_everything(0)

    train, val, test = get_loaders(
        experiment=experiment,
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        transforms_test=transforms_test,
        workers=workers,
        ds_frac=ds_frac,
    )
    if without is not None and without in ["alpha", "intermediate"]:
        model = ModelAblations(
            backbone=experiment["backbone"],
            nproj=experiment["nproj"],
            proj_dim=experiment["proj_dim"],
            without=without,
            device=device,
        )
    else:
        model = Model(
            backbone=experiment["backbone"],
            nproj=experiment["nproj"],
            proj_dim=experiment["proj_dim"],
            device=device,
        )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=experiment["lr"])
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    if without is None or without != "contrastive":
        supcon = SupConLoss()

    print(json.dumps(experiment, indent=2))
    results = {"val_loss": [], "val_acc": [], "test": {}}
    rlr = 0
    training_time = 0
    for epoch in range(max(epochss)):
        training_epoch_start = time.time()
        # Reduce learning rate
        if epoch + 1 in epochs_reduce_lr:
            rlr += 1
            optimizer.param_groups[0]["lr"] = experiment["lr"] / 10**rlr

        # Training
        model.train()
        for i, data in enumerate(train):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            optimizer.zero_grad()
            loss_ = bce(outputs[0], labels.float().view(-1, 1))
            if without is None or without != "contrastive":
                loss_ += experiment["factor"] * supcon(
                    F.normalize(outputs[1]).unsqueeze(1), labels
                )
            loss_.backward()
            optimizer.step()
            print(
                f"\r[Epoch {epoch + 1:02d}/{max(epochss):02d} | Batch {i + 1:04d}/{len(train):04d} | Time {training_time + time.time() - training_epoch_start:1.1f}s] loss: {loss_.item():1.4f}",
                end="",
            )
        training_time += time.time() - training_epoch_start

        # Validation
        model.eval()
        y_true = []
        y_score = []
        val_loss = 0
        with torch.no_grad():
            for data in val:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss_ = bce(outputs[0], labels.float().view(-1, 1))
                if without is None or without != "contrastive":
                    loss_ += experiment["factor"] * supcon(
                        F.normalize(outputs[1]).unsqueeze(1), labels
                    )
                val_loss += loss_.item()
                y_true.extend(labels.cpu().numpy().tolist())
                y_score.extend(
                    torch.sigmoid(outputs[0]).squeeze().cpu().numpy().tolist()
                )

        val_acc = accuracy_score(np.array(y_true), np.array(y_score) > 0.5)
        results["val_loss"].append(val_loss / len(val))
        results["val_acc"].append(val_acc)
        print(f", val_loss: {val_loss / len(val):1.4f}, val_acc: {val_acc:1.4f}")

        if epoch + 1 in epochss:
            # Testing
            accs = []
            aps = []
            print("generator: ACC / AP")
            for g, loader in test:
                model.eval()
                y_true = []
                y_score = []
                with torch.no_grad():
                    for data in loader:
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(
                            images
                            if experiment["training_set"] == "progan"
                            else images.view(-1, 3, 224, 224)
                        )
                        y_true.extend(labels.cpu().numpy().tolist())
                        if experiment["training_set"] == "progan":
                            y_score.extend(
                                torch.sigmoid(outputs[0]).cpu().numpy().tolist()
                            )
                        else:
                            y_score.extend(
                                torch.sigmoid(
                                    outputs[0]
                                    .view(images.shape[0], images.shape[1])
                                    .mean(1)
                                )
                                .cpu()
                                .numpy()
                                .tolist()
                            )

                test_acc = accuracy_score(np.array(y_true), np.array(y_score) > 0.5)
                test_ap = average_precision_score(y_true, y_score)
                accs.append(test_acc)
                aps.append(test_ap)

                results["test"][g] = {
                    "acc": test_acc,
                    "ap": test_ap,
                }

                print(f"{g}: {100 * test_acc:1.1f} / {100 * test_ap:1.1f}")

            print(
                f"Mean: {100 * sum(accs) / len(accs):1.1f} / {100 * sum(aps) / len(aps):1.1f}"
            )

            if store:
                ckpt_name = (
                    f"ckpt/model_{len(experiment['classes'])}class_trainable.pth"
                    if experiment["training_set"] == "progan"
                    else f"ckpt/model_ldm_trainable.pth"
                )
                print(f"Saving {ckpt_name} ...")
                torch.save(
                    {
                        k: model.state_dict()[k]
                        for k in model.state_dict()
                        if "clip" not in k
                    },
                    ckpt_name,
                )
            else:
                log = {
                    "epochs": epoch + 1,
                    "config": experiment,
                    "results": copy.deepcopy(results),
                }
                if without is None:
                    if experiment["training_set"] == "ldm":
                        filename = f'{experiment["savpath"]}_{epoch+1}.pickle'
                    elif epoch:
                        filename = f'{experiment["savpath"].replace("grid", "epochs")}_{epoch+1}.pickle'
                    elif ds_frac is not None:
                        filename = f'{experiment["savpath"].replace("grid", "dataset_size")}_{epoch+1}_{ds_frac}.pickle'
                    else:
                        filename = f'{experiment["savpath"]}_{epoch+1}.pickle'
                else:
                    filename = f"results/ablations/ncls_{len(experiment['classes'])}_{without}.pickle"
                with open(filename, "wb") as h:
                    pickle.dump(log, h, protocol=pickle.HIGHEST_PROTOCOL)


def get_our_trained_model(ncls, device):
    if ncls == 1:
        nproj = 4
        proj_dim = 1024
    elif ncls == 2:
        nproj = 4
        proj_dim = 128
    elif ncls == 4:
        nproj = 2
        proj_dim = 1024
    elif ncls == "ldm":
        nproj = 4
        proj_dim = 1024

    model = Model(
        backbone=("ViT-L/14", 1024),
        nproj=nproj,
        proj_dim=proj_dim,
        device=device,
    )
    setting = "ldm" if ncls == "ldm" else f"{ncls}class"
    ckpt_path = f"ckpt/model_{setting}_trainable.pth"
    state_dict = torch.load(ckpt_path, map_location=device)
    for name in state_dict:
        exec(
            f'model.{name.replace(".", "[", 1).replace(".", "].", 1)} = torch.nn.Parameter(state_dict["{name}"])'
        )
    return model


def get_generators(data="progan"):
    if data == "progan":
        return [
            "progan",
            "stylegan",
            "stylegan2",
            "biggan",
            "cyclegan",
            "stargan",
            "gaugan",
            "deepfake",
            "seeingdark",
            "san",
            "crn",
            "imle",
            "diffusion_datasets/guided",
            "diffusion_datasets/ldm_200",
            "diffusion_datasets/ldm_200_cfg",
            "diffusion_datasets/ldm_100",
            "diffusion_datasets/glide_100_27",
            "diffusion_datasets/glide_50_27",
            "diffusion_datasets/glide_100_10",
            "diffusion_datasets/dalle",
        ]
    elif data == "synthbuster":
        return [
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


# this function guarantees reproductivity
# other packages also support seed options, you can add to this function
def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ["PYTHONHASHSEED"] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluation(model, test, device, training="progan", ours=False, filename=None):
    accs = []
    aps = []
    log = {}
    for g, loader in test:
        model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for data in loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                if ours:
                    outputs = model(
                        images if training == "progan" else images.view(-1, 3, 224, 224)
                    )[0]
                else:
                    outputs = model(images)
                y_true.extend(labels.cpu().numpy().tolist())
                if training == "progan":
                    y_score.extend(torch.sigmoid(outputs).cpu().numpy().tolist())
                else:
                    y_score.extend(
                        torch.sigmoid(
                            outputs.view(images.shape[0], images.shape[1]).mean(1)
                        )
                        .cpu()
                        .numpy()
                        .tolist()
                    )

        test_acc = accuracy_score(np.array(y_true), np.array(y_score) > 0.5)
        test_ap = average_precision_score(y_true, y_score)
        accs.append(test_acc)
        aps.append(test_ap)
        log[g] = {
            "acc": test_acc,
            "ap": test_ap,
        }
        print(f"{g}: {100 * test_acc:1.1f} / {100 * test_ap:1.1f}")
    print(
        f"Mean: {100 * sum(accs) / len(accs):1.1f} / {100 * sum(aps) / len(aps):1.1f}"
    )
    if filename is not None:
        with open(filename, "wb") as h:
            pickle.dump(log, h, protocol=pickle.HIGHEST_PROTOCOL)


def data_augment(img):
    img = np.array(img)

    if random.random() < 0.5:
        sig = sample_continuous([0.0, 3.0])
        gaussian_blur(img, sig)

    if random.random() < 0.5:
        method = sample_discrete(["cv2", "pil"])
        qual = sample_discrete([30, 100])
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


def jpeg_from_key(img, compress_val, key):
    jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}
    method = jpeg_dict[key]
    return method(img, compress_val)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
"""
SID Codebase
"""

def extract_clip_features(
        experiment,
        dl=None,
        model=None,
        preprocess=None,
        split="train",
        ds_frac=0.1,
        device=None,
        target="both",
        save=False,
        workers=12,
):
    """
    Extract CLIP features for the dataset.
    """
    if model is None:
        model, preprocess = clip.load(experiment["backbone"][0], device=device)
    model.to(device)
    model.eval()    

    # Dataloader
    if dl is None:
        dl = get_loader(
            experiment=experiment,
            split=split,
            transforms=preprocess,
            workers=workers,
            ds_frac=ds_frac,
            target=target
        )
    
    features = []
    labels = []
    # Get the features
    with torch.no_grad():
        for data in tqdm.tqdm(dl, desc="Extracting features"):
            images, label = data
            images = images.to(device)
            feature = model.encode_image(images)
            features.append(feature.to(device))
            labels.append(label.to(device))
        features = torch.cat(features, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()

    if save:
        os.makedirs(f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/{target}", exist_ok=True)
        torch.save(features, f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/{target}/features.pt")
        print("Saved features to", f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/{target}/features.pt")

    return features, labels

def get_feature_loader(
        experiment, split, workers, ds_frac=None, target="both"
):
    classes = experiment["classes"] if split not in "test" else get_generators(experiment["training_set"])
    return DataLoader(
            FeatureDataset(
                split=split,
                classes=classes,
                ds_frac=ds_frac,
                target=target
            ),
            batch_size=experiment["batch_size"],
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=False
        )

def train_flow_experiment(
    experiment,
    epochss,
    epochs_reduce_lr,
    workers,
    device,
    store=False,
    ds_frac=None,
):
    seed_everything(0)

    train = get_feature_loader(experiment=experiment, split="train", workers=workers, ds_frac=ds_frac, target="real")
    val = get_feature_loader(experiment=experiment, split="val", workers=workers, ds_frac=ds_frac, target="both")
    test = get_feature_loader(experiment=experiment, split="test", workers=workers, ds_frac=ds_frac, target="both")
    input_dim = len(train.dataset.features[0][0])

    if experiment["flow"] in "nf":
        model = NormalizingFlow(
            input_dim=input_dim,
            num_steps=experiment["num_steps"]
        )
    elif experiment["flow"] in "glow":
        model = MiniGlow(
            input_dim=input_dim,
            num_steps=experiment["num_steps"]
        )
    else:
        raise ValueError("Flow type not supported")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=experiment["lr"])
    
    print(json.dumps(experiment, indent=2))
    results = {"val_loss": [], "val_acc": [], "test": {}}

    train_loss = []
    rlr = 0
    for epoch in range(max(epochss)):
        if epoch + 1 in epochs_reduce_lr:
            rlr += 1
            optimizer.param_groups[0]["lr"] = experiment["lr"] / 10**rlr
        
        model.train()

        with tqdm.tqdm(
                total=len(train),
                desc=f"Epoch {epoch + 1}/{max(epochss)}",
                unit="batch",
                ncols=100,
            ) as pbar:
            pbar.set_postfix({
                "loss": torch.inf
            })
            for data in train:
                features, _ = data
                features = features.float().to(device)
                loss = - model.log_prob(features).mean(axis=0)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({
                    "loss": loss.item()
                })
                pbar.update(1)
            pbar.close()
        
        # Validation
        model.eval()
        y_true = []
        y_score = []
        val_loss = 0

        with torch.no_grad():
            with tqdm.tqdm(
                total=len(val),
                desc="Validation",
                unit="batch",
                ncols=100
            ) as pbar:
                for data in val:
                    features, labels = data
                    features, labels = features.float().to(device), labels.to(device)
                    log_probs = model.log_prob(features)
                    y_pred = torch.exp(log_probs) < 0.005
                    val_loss -= log_probs.mean().item()
                    y_true.extend(labels.cpu().numpy().tolist())
                    y_score.extend(y_pred.cpu().numpy().tolist())
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": np.mean(val_loss)
                    })
                pbar.close()

        val_acc = accuracy_score(np.array(y_true), np.array(y_score))
        results["val_loss"].append(val_loss / len(val))
        results["val_acc"].append(val_acc)
        print(f"val_loss: {val_loss / len(val):1.4f}, val_acc: {val_acc:1.4f}")

    os.makedirs(f"{experiment['savpath']}", exist_ok=True)
    plt.plot(
        range(len(train_loss)),
        train_loss,
        label="Train Loss",
    )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(f"{experiment['savpath']}/train_loss.png")