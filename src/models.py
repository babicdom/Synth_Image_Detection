import torch
import torch.nn as nn
import clip
from src.nf import NormalizingFlow, MiniGlow

class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class Model(nn.Module):
    def __init__(
        self,
        backbone,
        nproj,
        proj_dim,
        device,
    ):
        super().__init__()

        self.device = device

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone[0], device=device)
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(backbone[1] if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            *[
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, 1),
            ]
        )

    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g_ = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]
        g = self.proj1(g_.float())

        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)

        return p, g_.sum(dim=1)


class FlowModel(nn.Module):
    def __init__(
        self,
        backbone,
        flow,
        n_steps,
        proj_dim,
        device,
    ):
        super().__init__()

        self.device = device

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone[0], device=device)
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(backbone[1] if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers).to(device)

        # Initialize the trainable part of the model
        self.flow = MiniGlow(input_dim=proj_dim, num_steps=n_steps) if flow in "glow" else NormalizingFlow(input_dim=backbone[1], num_steps=n_steps)
        self.flow.to(device)


    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]
        g = self.proj1(g.float()).sum(axis=1)
        p = self.flow.log_prob(g)
        return p
    
if __name__ == "__main__":
    # Example usage
    backbone = ("ViT-L/14", 1024)
    nproj = 2
    proj_dim = 512
    device = "cuda:0"

    flow_model = FlowModel(backbone, "glow", 4, 512, device)

    # Example input
    x = torch.randn(1, 3, 224, 224).to(device)
    output = flow_model(x)
    print("Output shape:", output.shape)