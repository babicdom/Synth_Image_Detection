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

for g, dl in test:
    print(f"Dataset {g} -")
    dl.dataset.get_sizes()
    if dl.dataset.real_size == dl.dataset.fake_size:
        print(f"Same sizes!")
    else:
        print(f"Real size: {dl.dataset.real_size}\nFake size: {dl.dataset.fake_size}")