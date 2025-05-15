import clip
from open_clip import create_model_from_pretrained
import timm
import torch
from torch.utils.data import DataLoader
import tqdm
from src.data import TrainingDatasetFreq
from sklearn.neighbors import KNeighborsClassifier
from src.utils import get_transform
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def get_model(model_name="ViT-L/14", device="cuda:0"):
    if model_name == "CLIP":
        model, _ = clip.load("ViT-L/14", device=device)
        tr_other = get_transform("val")
        tr_spec = get_transform("spec")
    elif model_name == "DinoV2":
        model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(model)
        tr_other = timm.data.create_transform(**data_config, is_training=False)
        tr_spec = get_transform("spec_dinov2")
    elif model_name == "SigLIP":
        model, pr = create_model_from_pretrained('hf-hub:timm/ViT-L-16-SigLIP2-256', device=device)
        tr_spec = get_transform("spec_siglip")
        tr_other = pr
    elif model_name == "ConvNextV2":
        model = timm.create_model('convnextv2_base.fcmae', pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(model)
        tr_other = timm.data.create_transform(**data_config, is_training=False)
        tr_spec = get_transform("spec_convnextv2")
    else:
        raise ValueError("Model not supported")
    model.to(device)
    return model, tr_other, tr_spec

def train_knn():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {}
    for model_name in ["CLIP", "DinoV2", "SigLIP", "ConvNextV2"]:
        model, tr_other, tr_spec = get_model(model_name, device)
        print(f"Running KNN for {model_name}")
        data = DataLoader(
            TrainingDatasetFreq(
                split="train", 
                classes= os.listdir(f"data/train"), # ["horse"], 
                transforms=[tr_other, tr_spec],
                ds_frac=0.1,
            ), 
            batch_size=32
        )
    
        model.eval()
        x = []
        y = []
        with torch.no_grad():
            for batch in tqdm.tqdm(data):
                im, label = batch
                im = im.to(device)
                if any([model_name == "CLIP", model_name == "SigLIP"]):
                    feat = model.encode_image(im)
                else:
                    feat = model(im)
                x.extend(feat.cpu().numpy().tolist())
                y.extend(label.cpu().numpy().tolist())

        acc = []
        for k in [1, 5, 10, 20, 50, 100]:
            print(f"n_neighbors: {k}")
            knn = KNeighborsClassifier(
                n_neighbors=k
            )
            knn.fit(X=x, y=y)
            preds = knn.predict(x)
            acc.append(accuracy_score(y, preds))
            print(f"Accuracy: {accuracy_score(y, preds)}")
        results[model_name] = acc
        tsne = TSNE(n_components=2, random_state=0)
        x = tsne.fit_transform(np.array(x))
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Feature Space")
        plt.savefig(f"results/frequency/feature_space_{model_name}.png")

    plt.figure(figsize=(10, 5))
    plt.title("KNN Accuracy")
    plt.xlabel("n_neighbors")
    plt.ylabel("Accuracy")
    plt.xticks([1, 5, 10, 20, 50, 100])
    for model_name, acc in results.items():
        plt.plot([1, 5, 10, 20, 50, 100], acc, label=model_name)
    plt.legend()
    plt.savefig("results/frequency/knn_accuracy.png")

train_knn()