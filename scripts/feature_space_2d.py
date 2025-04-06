from scripts.results import plot_2d_feature_space, get_feaure_space, get_trained_model
import os
import pickle
import numpy as np
from sklearn.manifold import TSNE

if __name__ == "__main__":
    dpi = 400

    # t-SNE feature spaces
    device = "cuda:0"
    for name in ["ufd", "rine"]:
        print(f"t-SNE feature space for {name}")
        if os.path.exists(f"results/figs/data_2d_{name}.pickle"):
            with open(f"results/figs/data_2d_{name}.pickle", "rb") as h:
                data_2d, labels_, gens_ = pickle.load(h)
        else:
            model = get_trained_model(name, device)
            data_, labels_, gens_ = get_feaure_space(
                model=model,
                batch_size=100,
                max_samples_per_gen=500,
                device=device,
                name=name,
            )
            data_2d = TSNE(
                n_components=2,
                learning_rate=10,
                init="pca",
                perplexity=30,
                n_iter=5000,
                random_state=0,
            ).fit_transform(np.concatenate(data_, axis=0))
            with open(f"results/figs/data_2d_{name}.pickle", "wb") as h:
                pickle.dump([data_2d, labels_, gens_], h)

        plot_2d_feature_space(
            data_2d,
            labels_,
            gens_,
            filename=f"results/figs/feature_space_{name}.png",
            dpi=dpi,
        )
