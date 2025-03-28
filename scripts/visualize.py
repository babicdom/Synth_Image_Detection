import clip
from src.utils import get_loaders, get_transforms
import os
import torch


MAIN_DIR = "/mnt/personal/babicdom"

def load_clip(backbone, device):
    """
    Load the CLIP model and its preprocess function.
    """
    # Load CLIP
    clip_model, preprocess = clip.load(backbone[0], device=device)
    
    # Freeze CLIP model parameters
    for param in clip_model.parameters():
        param.requires_grad = False

    return clip_model, preprocess

def visualize_features_tsne(experiment, ds_frac=0.1, device='cpu'):
    """
    Visualize the features of the dataset using the CLIP model.
    """

    transforms_train, _, transforms_test = get_transforms()
    # Get a batch of images and labels
    print('Loading dataloader')
    train, _, test = get_loaders(
        experiment=experiment,
        transforms_train=transforms_train,
        transforms_test=transforms_test,
        transforms_val=None,
        ds_frac=ds_frac,
        workers=12
    )
    print('Finished loading daatloader')
    # Get the CLIP model and preprocess function
    model, preprocess = load_clip(backbone, device)

    print('Finished loading model')

    features = []
    labels = []
    # Get the features
    with torch.no_grad():
        for data in train:
            images, label = data
            images = images.to(device)
            features.append(model.encode_image(images).cpu())
            labels.append(label.cpu())
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    # Convert to numpy arrays
    features = features.numpy()
    labels = labels.numpy()

    print(features.shape)

if __name__ == "__main__":
    # Example usage
    experiment = {
        "training_set": "progan",
        "backbone": ("ViT-L/14", 1024),
        "classes": os.listdir(f"{MAIN_DIR}/data/train/"),
        "batch_size": 64
    }
    print("Visualizing features with t-SNE")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    backbone = ("ViT-B/32", 768)
    visualize_features_tsne(experiment=experiment)