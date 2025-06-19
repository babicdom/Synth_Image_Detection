import torch
from src.spai import build_mf_vit, build_finetune_optimizer, build_scheduler, load_pretrained
from src.utils import get_transform, get_loaders, train_spai as train
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os

data_path = "weights/spai.pth"
checkpoint = torch.load(data_path, map_location='cpu', weights_only=False)
config = checkpoint["config"]
print(data_path)

experiment = {
    # Training based
    "crop": 224, # 256, # 
    "imgsize": 256,
    "training_set": "progan",
    "contrastive": True,
    "batch_size": 64,
    "classes": os.listdir(f"data/train"), # ["horse", 'chair', 'car', 'cat'], # 
    "save_path": "SPAI",
    "window_slide":True,
    "train":config.TRAIN
}
# Set a fixed seed to all the random number generators.
seed = config.SEED
torch.manual_seed(seed)
np.random.seed(seed)
# random.seed(seed)
cudnn.benchmark = True

transforms_train = get_transform(split="train_spai")
transforms_val = get_transform(split="no_crop_spai")
train_, val, test = get_loaders(
    experiment=experiment,
    transforms_train=transforms_train,
    transforms_val=transforms_val,
    transforms_test=transforms_val, # transforms_test,
    workers=2,
)

model = build_mf_vit(config)
model.cuda()

optimizer = build_finetune_optimizer(config, model)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"number of params: {n_parameters}")
if hasattr(model, 'flops'):
    flops = model.flops()
    print(f"number of GFLOPs: {flops / 1e9}")

lr_scheduler = build_scheduler(config, optimizer, len(train_))
criterion: nn.Module = lambda x: nn.BCEWithLogitsLoss()(x.squeeze())

if config.PRETRAINED:
    load_pretrained(config, model.get_vision_transformer())
else:
    model.unfreeze_backbone()
    print(f"No pretrained model. Backbone parameters are trainable.")

train(
    experiment=experiment,
    model=model,
    data=[train_, val, test],
    loss_fn=criterion,
    optimizer=optimizer,
    scheduler=lr_scheduler,
    epochs=8,
    device="cuda:0",
    store=True    
)