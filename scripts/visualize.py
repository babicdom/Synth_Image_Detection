from src.utils import extract_clip_features
import os
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

MAIN_DIR = "/mnt/personal/babicdom"

def visualize_features_tsne(experiment, split="train", ds_frac=0.1, device=None):
    """
    Visualize the features of the dataset using the CLIP model.
    """
    real_features, _ = extract_clip_features(
        experiment=experiment,
        split=split,
        ds_frac=ds_frac,
        device=device,
        target="real",
    )

    fake_features, _ = extract_clip_features(
        experiment=experiment,
        split=split,
        ds_frac=ds_frac,
        device=device,
        target="fake",
    )

    if not os.path.exists(f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}"):
        os.makedirs(f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/0_real")
        os.makedirs(f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/1_fake")
  
    # Save the features and labels
    torch.save(real_features, f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/0_real/features.pt")
    print("Saved real features to", f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/0_real/features.pt")
    torch.save(fake_features, f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/1_fake/features.pt")
    print("Saved fake features to", f"{experiment['featpath']}/{split}/{'_'.join(experiment['classes'])}/1_fake/features.pt")

    # Get the t-SNE representation
    tsne = TSNE(n_components=2).fit(real_features)
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
    if not os.path.exists(f"{experiment['savpath']}_{'_'.join(experiment['classes'])}"):
        os.makedirs(f"{experiment['savpath']}_{'_'.join(experiment['classes'])}")
  
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
        device=device, 
        split="train", 
        ds_frac=1
        )