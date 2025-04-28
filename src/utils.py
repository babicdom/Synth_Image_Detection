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
from einops import rearrange

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, confusion_matrix

from src.data import TrainingDataset, TrainingDatasetLDM, EvaluationDataset, FeatureDataset

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
    experiment, split, transforms, workers, target="both"
):
    if experiment["training_set"] == "progan":
        if split == "train":
            return DataLoader(
                    TrainingDataset(
                        split="train",
                        classes=experiment["classes"],
                        transforms=transforms,
                        ds_frac=experiment.get("ds_frac", None),
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
                        split="val", 
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
            for g in get_generators()
        ]
    else:
        raise ValueError("split must be one of train, val, test")
    
def get_loaders(
    experiment, transforms_train, transforms_val, transforms_test, workers, target="both"
):
    train = get_loader(
        experiment,
        split="train",
        transforms=transforms_train,
        workers=workers,
        target=target
    )
    val = get_loader(
        experiment,
        split="val",
        transforms=transforms_val,
        workers=workers,
    )
    test = get_loader(
        experiment,
        split="test",
        transforms=transforms_test,
        workers=workers,
    )
    return train, val, test


def get_generators():
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
        "whichfaceisreal",
        "diffusion_datasets/guided",
        "diffusion_datasets/ldm_200",
        "diffusion_datasets/ldm_200_cfg",
        "diffusion_datasets/ldm_100",
        "diffusion_datasets/glide_100_27",
        "diffusion_datasets/glide_50_27",
        "diffusion_datasets/glide_100_10",
        "diffusion_datasets/dalle",
        "synthbuster/dalle2",
        "synthbuster/dalle3",
        "synthbuster/stable-diffusion-1-3",
        "synthbuster/stable-diffusion-1-4",
        "synthbuster/stable-diffusion-2",
        "synthbuster/stable-diffusion-xl",
        "synthbuster/glide",
        "synthbuster/firefly",
        "synthbuster/midjourney-v5",
        "flux",
        "gigagan",
        "midjourney-v6.1",
        "stable-diffusion-3",

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
        use_transform=True,
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
        if use_transform:
            transforms_ = get_transform(split)
        else:
            transforms_ = preprocess
        # Get the dataloader
        dl = get_loader(
            experiment=experiment,
            split=split,
            transforms=transforms_,
            workers=workers,
            ds_frac=ds_frac,
            target=target
        )
    
    features = []
    # Get the features
    with torch.no_grad():
        for data in tqdm.tqdm(dl, desc="Extracting features"):
            images, _ = data
            images = images.to(device)
            if split == "test":
                images = images.view(-1, 3, 224, 224)
            feature = model.encode_image(images)
            features.append(feature.to(device))
        features = torch.cat(features, dim=0).cpu()

    if save:
        os.makedirs(f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/{target}", exist_ok=True)
        torch.save(features, f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/{target}/features.pt")
        print("Saved features to", f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/{target}/features.pt")

    return features

def get_feature_loader(
        experiment, split, workers, ds_frac=None, target="both"
):
    if split in "test":
        return [
            (g, DataLoader(
                    FeatureDataset(
                    split=split,
                    classes=[g],
                    ds_frac=ds_frac,
                    target=target
                    ),
                    batch_size=experiment["batch_size"],
                    shuffle=True,
                    num_workers=workers,
                    pin_memory=True,
                    drop_last=False
                )) for g in get_generators(experiment["training_set"])
        ]
    else:
        return DataLoader(
                FeatureDataset(
                    split=split,
                    classes=experiment["classes"],
                    ds_frac=ds_frac,
                    target=target
                ),
                batch_size=experiment["batch_size"],
                shuffle=True,
                num_workers=workers,
                pin_memory=True,
                drop_last=False
            )

def find_best_acc_threshold(y_true, y_pred):
    thresholds = np.linspace(0, 1, 100)
    best_accuracy = 0
    best_threshold = 0

    for threshold in thresholds:
        accuracy = accuracy_score(y_true, y_pred > threshold)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold

def calculate_for_threshold(y_true, y_pred, threshold):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > threshold)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > threshold)
    acc = accuracy_score(y_true, y_pred > threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred > threshold).ravel()

    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0  # Negative Predictive Value
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # Recall (True Positive Rate)
    tnr = tn / (fp + tn) if (fp + tn) != 0 else 0  # True Negative Rate

    return { 
        'r_acc': r_acc, 'f_acc': f_acc, 'acc': acc, 
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
        'ppv': ppv, 'npv': npv, 'tpr': tpr, 'tnr': tnr 
    }  

bce = nn.BCEWithLogitsLoss(reduction="sum")
supcon = SupConLoss()
def transformer_train_loss(factor, contrastive, unsqueeze=False):
    def _transformer_train_loss(output, labels):
        if unsqueeze:
            loss_ = bce(output[0], labels.unsqueeze(1).repeat(1, output[0].shape[1]))
        else:
            loss_ = bce(output[0], labels.float())
        if contrastive:
            loss_ += factor * supcon(
                F.normalize(output[-1]).unsqueeze(1), labels
            )
        return loss_
    return _transformer_train_loss

def train(
    experiment,
    model,
    data=None,
    loss_fn=bce,
    optimizer=None,
    scheduler=None,
    epochs=10,
    workers=12,
    device="cpu",
    score_fn=lambda x:torch.sigmoid(x).squeeze(),
    store=False,
):
    seed_everything(0)

    if data is None:
        transforms_train, transforms_val, transforms_test = get_transforms()
        train, val, test = get_loaders(
            experiment=experiment,
            transforms_train=transforms_train,
            transforms_val=transforms_val,
            transforms_test=transforms_val, # transforms_test,
            workers=workers,
        )
    else:
        train, val, test = data
    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=experiment["lr"])
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=experiment["lr_step"],
            gamma=experiment["lr_gamma"],
        )
    
    print(json.dumps(experiment, indent=2))
    results = {"val_loss": [], "val_ap": [], "val_auc": [], "test": {}}

    train_loss = []
    for epoch in range(epochs):
        model.train()

        with tqdm.tqdm(
                total=len(train),
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                ncols=100,
            ) as pbar:
            pbar.set_postfix({
                "loss": torch.inf
            })
            for data in train:
                # prev_model = model.copy()
                images, labels = data
                images, labels = images.float().to(device), labels.float().to(device)
                # try:
                loss = loss_fn(model(images), labels)
                #except Exception as e:
                    # for layer in prev_model.flow.named_parameters():
                    #     print(layer)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    images, labels = data
                    images, labels = images.float().to(device), labels.float().to(device)
                    output = model(images)
                    val_loss = loss_fn(output, labels)
                    scores = score_fn(output)
                    y_true.extend(labels.cpu().numpy().tolist())
                    y_score.extend(scores.cpu().numpy().tolist())
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": val_loss.item()
                    })
                pbar.close()
    
        val_ap = average_precision_score(y_true, y_score)
        val_auc = roc_auc_score(y_true, y_score)
        results["val_ap"].append(val_ap)
        results["val_auc"].append(val_auc)
        print(f"val_ap: {val_ap:1.4f}, val_auc: {val_auc:1.4f}")
        scheduler.step()

    if store:
        os.makedirs(f"ckpt/{experiment["save_path"]}/", exist_ok=True)
        ckpt_name = f"ckpt/{experiment["save_path"]}/train.pth"
        print(f"Saving {ckpt_name} ...")
        torch.save(model.state_dict(), ckpt_name)
        pickle.dump(
            experiment,
            open(f"ckpt/{experiment["save_path"]}/experiment.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    log = {
        "epochs": epoch + 1,
        "config": experiment,
        "results": copy.deepcopy(results),
    }
    os.makedirs(f"results/{experiment["save_path"]}/", exist_ok=True)
    filename = f"results/{experiment["save_path"]}/train.pickle"
    with open(filename, "wb") as h:
        pickle.dump(log, h, protocol=pickle.HIGHEST_PROTOCOL)

    # Testing
    eval_model(
        experiment=experiment,
        model=model,
        test=test,
        score_fn=score_fn,
        device=device
        )

def eval_model(experiment, model, score_fn, test=None, device="cuda:0"):
    results = {}
    aps = []
    aucs = []
    accs = []

    if test is None:
        transform = get_transform("val")
        test = get_loader(
            experiment=experiment,
            split="test",
            transforms=transform,
            workers=12,
        )

    print("Testing - generator: ACC / AP / AUC")
    for g, dl in test:
        model.eval()
        y_true = []
        y_score = []

        print(f'Fake: {len(dl.dataset.fake)}, Real: {len(dl.dataset.real)}, Total: {len(dl.dataset.images)}')
        with torch.no_grad():
            for data in tqdm.tqdm(dl, desc=f"Testing on generator {g}", unit="batch"):
                images, labels = data
                images, labels = images.float().to(device), labels.to(device)
                output = model(images)
                scores = score_fn(output)
                y_true.extend(labels.cpu().numpy().tolist())
                y_score.extend(scores.cpu().numpy().tolist())

        test_ap = average_precision_score(y_true, y_score)
        test_auc = roc_auc_score(y_true, y_score)
        threshold = find_best_acc_threshold(np.array(y_true), np.array(y_score))
        threshold_acc = calculate_for_threshold(np.array(y_true), np.array(y_score), threshold)

        aps.append(test_ap)
        aucs.append(test_auc)
        accs.append(threshold_acc["acc"])

        results[g] = {
            "ap": test_ap,
            "auroc": test_auc,
            "acc": threshold_acc["acc"],
            "tpr": threshold_acc["tpr"],
            "tnr": threshold_acc["tnr"],
        }
        print(f"{g}: {100 * threshold_acc["acc"]:1.2f} / {100 * test_ap:1.2f} / {100 * test_auc:1.2f}")

    print(
        f"Mean: {100 * sum(accs) / len(accs):1.2f} / {100 * sum(aps) / len(aps):1.2f} / {100 * sum(aucs) / len(aucs):1.2f}"
    )

    log = {
        "config": experiment,
        "results": copy.deepcopy(results),
    }
    filename = f"results/{experiment["save_path"]}/eval.pickle"
    with open(filename, "wb") as h:
        pickle.dump(log, h, protocol=pickle.HIGHEST_PROTOCOL)