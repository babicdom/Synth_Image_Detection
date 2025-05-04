import clip
import torch
from torch.utils.data import DataLoader
import tqdm
from src.data import TrainingDatasetFreq
from sklearn.neighbors import KNeighborsClassifier
from src.utils import get_transform
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

tr_other = get_transform("other")
tr_spec = get_transform("spec")
data = DataLoader(
    TrainingDatasetFreq(
        split="train", 
        classes= os.listdir(f"data/train"), # ["horse"], 
        transforms=[tr_other, tr_spec],
        ds_frac=0.2,
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
        feat = model.encode_image(im)
        x.extend(feat.cpu().numpy().tolist())
        y.extend(label.cpu().numpy().tolist())
        del im, label
        torch.cuda.empty_cache()

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
plt.plot([1, 5, 10, 20, 50, 100], acc)
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy")
plt.savefig("results/frequency/test.png")
plt.show()
