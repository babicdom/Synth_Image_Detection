from src.utils import get_clip_features
import os
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

MAIN_DIR = "/mnt/personal/babicdom"

def visualize_features_tsne(experiment, mode="load", split="train", ds_frac=0.1, device=None, save=True):
    """
    Visualize the features of the dataset using the CLIP model.
    """
    real_features, _ = get_clip_features(
        mode=mode,
        experiment=experiment,
        split=split,
        ds_frac=ds_frac,
        device=device,
        target="real",
        save=save,
    )

    fake_features, _ = get_clip_features(
        mode=mode,
        experiment=experiment,
        split=split,
        ds_frac=ds_frac,
        device=device,
        target="fake",
        save=save,
    )

    print("Real features shape:", real_features.shape)
    print("Fake features shape:", fake_features.shape)
    # Get the t-SNE representation
    tsne = TSNE(n_components=2, verbose=1).fit(real_features)
    real_images = tsne.transform(real_features)
    fake_images = tsne.transform(fake_features)

    # Plot the t-SNE representation
    plt.figure(figsize=(10, 10))
    plt.scatter(real_images[:, 0], real_images[:, 1], c='blue', alpha=0.5, label='real')
    plt.scatter(fake_images[:, 0], fake_images[:, 1], c='red', alpha=0.5, label='fake')
    plt.legend(loc='upper right', fontsize=10)
    plt.title("t-SNE visualization of CLIP features")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    
    # Saving plots and features
    os.makedirs(f"{experiment['savpath']}_{'_'.join(experiment['classes'])}", exist_ok=True)
  
    # Save the t-SNE plot
    plt.savefig(f"{experiment['savpath']}_{'_'.join(experiment['classes'])}/tsne.png")    
    plt.close()
    print("Saved t-SNE plot to", f"{experiment['savpath']}_{'_'.join(experiment['classes'])}/tsne.png")

if __name__ == "__main__":
    experiment = {
        "training_set": "progan",
        "backbone": ("ViT-L/14", 1024),
        "classes": os.listdir(f"{MAIN_DIR}/data/train/"),
        "batch_size": 32,
        "savpath": f"results/visualize/tsne",
        "featpath": "results/features",
    }
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    visualize_features_tsne(
        experiment=experiment,
        mode="load",
        device=device, 
        split="train", 
        ds_frac=1,
        save=True,
        )