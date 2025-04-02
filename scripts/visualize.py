from src.utils import get_clip_features
import os
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

MAIN_DIR = "/mnt/personal/babicdom"

def visualize_features_tsne(experiment, split="train"):
    """
    Visualize the features of the dataset using the CLIP model.
    """
    real_features = torch.cat(
            [
                torch.load(f"{experiment["featpath"]}/{split}/{y}/real/features.pt")
                for y in experiment["classes"]
            ]
        )

    fake_features = torch.cat(
            [
                torch.load(f"{experiment["featpath"]}/{split}/{y}/fake/features.pt")
                for y in experiment["classes"]
            ]
        )

    print("Real features shape:", real_features.shape)
    print("Fake features shape:", fake_features.shape)
    # Concatenate the features
    combined_features = torch.cat((real_features, fake_features), dim=0)
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)
    print("Scaled combined features shape:", combined_features.shape)

    # Apply PCA
    pca_features = PCA(n_components=50).fit_transform(combined_features)
    print("PCA features shape:", pca_features.shape)
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=3)
    tsne_results = tsne.fit_transform(combined_features)

    # Plot the t-SNE representation
    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_results[:len(real_features), 0], real_features[:len(real_features), 1], c='blue', alpha=0.5, label='real')
    plt.scatter(tsne_results[len(real_features):, 0], tsne_results[len(real_features):, 1], c='red', alpha=0.5, label='fake')
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
        "classes": ["horse", "chair"], # os.listdir(f"{MAIN_DIR}/data/train/"),
        "savpath": f"results/visualize/tsne",
        "featpath": "results/features",
    }
    visualize_features_tsne(
        experiment=experiment,
        split="train", 
        )