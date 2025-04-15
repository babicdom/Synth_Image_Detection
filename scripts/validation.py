from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import EvaluationDataset
from src.utils import (
    get_transforms,
    get_our_trained_model,
    get_generators,
    evaluation,
)

device = "cuda:0"
generators = get_generators()
_, transforms, _ = get_transforms()

test = [
    (
        g,
        DataLoader(
            EvaluationDataset(g, transforms=transforms),
            batch_size=128,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            drop_last=False,
        ),
    )
    for g in generators
]

generators = get_generators(data="synthbuster")
_, _, transforms = get_transforms()
test_ldm = [
    (
        g,
        DataLoader(
            EvaluationDataset(g, transforms=transforms),
            batch_size=16,
            shuffle=False,
            num_workers=12,
            pin_memory=True,
            drop_last=False,
        ),
    )
    for g in generators
]

for ncls in [1, 2, 4, "ldm"]:
    print(f"\n{ncls}-class")
    model = get_our_trained_model(ncls=ncls, device=device)
    model.to(device)
    evaluation(model, test, device, ours=True)
    evaluation(model, test_ldm, device, training="ldm", ours=True)