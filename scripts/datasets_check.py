from src.utils import patchify_image, get_transform, get_loader, image_enlisting_collate_fn
import pickle


experiment = pickle.load(
    open(f"ckpt/IntermediatePatch/3_nproj_512_proj_dim/experiment.pickle", "rb")
)
tr = get_transform("val")
test = get_loader(
    experiment=experiment,
    split="test",
    transforms=tr,
)

for g, dataset in test:
    print(f"Dataset {g} -")
    for _, _, _ in dataset:
        continue

    print(f"Real size: {dataset.dataset.real_size}\nFake size: {dataset.dataset.fake_size}")