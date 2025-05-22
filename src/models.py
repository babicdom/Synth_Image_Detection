from functools import partial
from typing import Callable
import torch
import torch.nn as nn
import clip
from src.nf import NormalizingFlow, MiniGlow
from src.utils import patchify_image
from open_clip import create_model_from_pretrained
from src.vision_transformer import Encoder
from einops import rearrange
import pickle
import timm

CLIP_SEQ_LENGTH=256

class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

class WindowedIntermediatePatch(nn.Module):
    def __init__(
        self,
        att_dim,
        n_heads,
        device,
        patch_size: tuple = (224, 224),
        stride: tuple = (16, 16),
        pooling = "max",
    ):
        super().__init__()

        self.device = device
        self.patch_size = patch_size
        self.stride = stride

        opt = pickle.load(
            open(f"ckpt/IntermediatePatch/3_nproj_512_proj_dim/experiment.pickle", "rb")
        )
        self.intermediate_patch = IntermediatePatch(
            backbone=opt["backbone"],
            nproj=opt["nproj"],
            proj_dim=opt["proj_dim"],
            device=torch.device("cuda:0"),
        )
        self.intermediate_patch.load_state_dict(
            torch.load(f"ckpt/IntermediatePatch/3_nproj_512_proj_dim/train.pth", map_location="cuda:0")
        )

        for param in self.intermediate_patch.parameters():
            param.requires_grad = False

        self.pooling = pooling

        self.window_attention = PatchAttention(
            att_dim=att_dim,
            n_heads=n_heads,
            hidden_dim=opt["proj_dim"],
        )
        
        self.to(device)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            out = self.forward_batch(x)
        elif isinstance(x, list):
            out = self.forward_list(x)
        else:
            raise ValueError("Input must be a tensor or a list of tensors")
        return out
            

    def forward_batch(self, x):
        x = patchify_image(x, self.patch_size, self.stride)

        with torch.no_grad():
            p, z = self.intermediate_patch(x)
            if self.pooling == "max":
                values, indices = p.max(dim=1)
                p = values
                z = z.gather(1, indices.unsqueeze(-1).expand(-1, -1, z.size(-1)))
            elif self.pooling == "mean":
                p = p.mean(dim=1)
                z = z.mean(dim=1)
            else:
                raise ValueError("Pooling method not supported")

        g = self.window_attention(z.permute(1, 0, 2))
        p = p * g
        p = p.sum(dim=1)
        return p, g
    
    def forward_list(self, x):
        x = [patchify_image(xi, self.patch_size, self.stride) for xi in x]
        patch_num = x[0].shape[1]

        with torch.no_grad():
            p, z = self.intermediate_patch(x)
            if self.pooling == "max":
                values, indices = p.max(dim=1)
                p = values
                z = z.gather(1, indices.unsqueeze(-1).expand(-1, -1, z.size(-1)))
            elif self.pooling == "mean":
                p = p.mean(dim=1)
                z = z.mean(dim=1)
            else:
                raise ValueError("Pooling method not supported")

        g = self.window_attention(z.permute(1, 0, 2))
        p = p * g
        p = p.sum(dim=1)
        return p, g
    
    def predict(
            self, 
            x: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            o, _ = self.forward(x)
            return o.sigmoid().flatten().cpu().numpy()

class AttentionIntermediatePatch(nn.Module):
    def __init__(
        self,
        att_dim,
        n_heads,
        device,
    ):
        super().__init__()

        self.device = device

        opt = pickle.load(
            open(f"ckpt/IntermediatePatchLDM_SupConLoss/3_nproj_512_proj_dim/experiment.pickle", "rb")
        )
        self.intermediate_patch = IntermediatePatch(
            backbone=opt["backbone"],
            nproj=opt["nproj"],
            proj_dim=opt["proj_dim"],
            device=torch.device("cuda:0"),
        )
        self.intermediate_patch.load_state_dict(
            torch.load(f"ckpt/IntermediatePatchLDM_SupConLoss/3_nproj_512_proj_dim/train.pth", map_location="cuda:0")
        )

        for name, param in self.intermediate_patch.named_parameters():
            param.requires_grad = False

        self.window_attention = PatchAttention(
            att_dim=att_dim,
            n_heads=n_heads,
            hidden_dim=opt["proj_dim"],
        )
        
        self.to(device)

    def forward(self, x):
        with torch.no_grad():
            p, z = self.intermediate_patch(x)
        g = self.window_attention(z.permute(1, 0, 2))
        p = p * g
        p = p.sum(dim=1)
        return p, g
    
    def predict(
            self, 
            x: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            o, _ = self.forward(x)
            return o.sigmoid().flatten().cpu().numpy()

class IntermediatePatch(nn.Module):
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
        # for name, param in self.clip.named_parameters():
        #     param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([256, 1, len(self.hooks), proj_dim])) # first dim is number of tokens, second is batch size, third is number of layers, last is projection dim
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
        self.to(device)

    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[1:, :, :, :]
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=2) * g
        z = torch.sum(z, dim=2)
        z = self.proj2(z)

        p = self.head(z).squeeze()
        if p.dim() == 2:
            p = p.permute(1, 0)
        return p, z
    
    def forward_with_grad(self, x):
        self.clip.encode_image(x)
        g = torch.stack([h.output for h in self.hooks], dim=2)[1:, :, :, :]
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=2) * g
        z = torch.sum(z, dim=2)
        z = self.proj2(z)

        p = self.head(z).squeeze()
        if p.dim() == 2:
            p = p.permute(1, 0)
        return p, z
    
    def predict(
            self, 
            x: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            o, _ = self.forward(x)
            if kwargs["method"] == "mean":
                return o.sigmoid().mean(-1).flatten().cpu().numpy()
            elif kwargs["method"] == "max":
                return o.sigmoid().max(-1).values.flatten().cpu().numpy()
            elif kwargs["method"] == "patchify":
                pass
            else:
                raise ValueError("Method not supported")

class PatchAttentionPool(nn.Module):
    def __init__(
            self, 
            att_dim: int,
            n_heads: int,
            dropout: int,
            hidden_dim: int,
        ):
        super().__init__()
        dim_head: int = att_dim // n_heads
        self.heads = n_heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.kv = nn.Linear(hidden_dim, att_dim*2, bias=False)
        self.patch_aggregator = nn.Parameter(torch.zeros((n_heads, 1, att_dim//n_heads)))
        nn.init.trunc_normal_(self.patch_aggregator, std=.02)
        self.o = nn.Sequential(
            nn.Linear(att_dim, hidden_dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
            self, 
            x: torch.Tensor,
            return_attn: bool = False,
    ):
        aggregator: torch.Tensor = self.patch_aggregator.expand(x.size(0), -1, -1, -1)
        kv = self.kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        dots = torch.matmul(aggregator, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.o(x)
        x = x.squeeze(dim=1)
        if return_attn:
            return x, attn
        else:
            return x
        
class PatchAttention(nn.Module):
    def __init__(
            self, 
            att_dim: int,
            n_heads: int,
            hidden_dim: int,
            dropout: int = 0.0,
        ):
        super().__init__()
        dim_head: int = att_dim // n_heads
        self.heads = n_heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.k = nn.Linear(hidden_dim, att_dim, bias=False)
        self.patch_aggregator = nn.Parameter(torch.zeros((n_heads, 1, att_dim//n_heads)))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.patch_aggregator, std=.02)

    def forward(
            self, 
            x: torch.Tensor,
    ):
        aggregator: torch.Tensor = self.patch_aggregator.expand(x.size(0), -1, -1, -1)
        k = self.k(x)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(aggregator, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        if self.heads > 1:
            attn = attn.mean(dim=1)
        attn = attn.squeeze()
        return attn


### Less important models

class CLIPatch(nn.Module):
    def __init__(
        self,
        backbone,
        device,
        n_layers: int,
        n_heads: int,
        mlp_dim: int,
        num_classes: int = 1,
        cls_ratio: int = 1,
        cls_dropout: float = 0.5,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.device = device

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone[0], device=device)
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hook to get the last layer tokens
        self.hook = Hook("transformer.resblocks.23.ln_2", self.clip.visual.transformer.resblocks[-1].ln_2)

        # Extension
        hidden_dim = backbone[1]
        self.encoder = Encoder(
            seq_length=CLIP_SEQ_LENGTH,
            num_layers=n_layers,
            num_heads=n_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

        # Classification head
        self.num_classes = num_classes
        self.cls = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*cls_ratio),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(hidden_dim*cls_ratio, hidden_dim*cls_ratio),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(hidden_dim*cls_ratio, num_classes)
        )
        self.to(device)
        
    def forward(
            self, 
            x: torch.Tensor
    ):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = self.hook.output[1:, :, :]
        g = g.permute(1, 0, 2)
        g = self.encoder(g)

        batch_size, num_patches, embedding_dim = g.shape
        g_reshaped = g.reshape(-1, embedding_dim).float()
        out_flat = self.cls(g_reshaped)
        
        if self.num_classes == 1:
            out = out_flat.reshape(batch_size, num_patches)
        else:
            out = out_flat.reshape(batch_size, num_patches, self.num_classes)    
        return out, g
    
    def predict(
            self, 
            x: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            o, _ = self.forward(x)
            if kwargs["method"] == "mean":
                return o.sigmoid().mean(-1).flatten().cpu().numpy()
            elif kwargs["method"] == "max":
                return o.sigmoid().max(-1).values.flatten().cpu().numpy()
            
class SigLIPIntermediate(nn.Module):
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
        self.siglip, self.preprocess = create_model_from_pretrained(backbone[0], device=device) # 'hf-hub:timm/ViT-L-16-SigLIP2-256', device=device)
        for name, param in self.siglip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.siglip.visual.named_modules()
            if "ls2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([256, 1, len(self.hooks), proj_dim]))
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
        self.to(device)

    def forward(self, x):
        with torch.no_grad():
            self.siglip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)
        g = g.permute(1, 0, 2, 3)
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=2) * g
        z = torch.sum(z, dim=2)
        z = self.proj2(z)

        p = self.head(z).squeeze()
        if p.dim() == 2:
            p = p.permute(1, 0)
        return p, z
    
    def predict(
            self, 
            x: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            o, _ = self.forward(x)
            if kwargs["method"] == "mean":
                return o.sigmoid().mean(-1).flatten().cpu().numpy()
            elif kwargs["method"] == "max":
                return o.sigmoid().max(-1).values.flatten().cpu().numpy()
            

class ConvNextV2Intermediate(nn.Module):
    def __init__(
        self,
        backbone,
        nproj,
        proj_dim,
        device,
    ):
        super().__init__()

        self.device = device

        # Load and freeze convnext
        self.convnext = timm.create_model('convnextv2_base.fcmae', pretrained=True, num_classes=0)
        for name, param in self.convnext.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook("", block)
            for st in self.convnext.stages
            for block in st.blocks
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([256, 1, len(self.hooks), proj_dim]))
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
        self.to(device)

    def forward(self, x):
        with torch.no_grad():
            self.siglip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)
        g = g.permute(1, 0, 2, 3)
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=2) * g
        z = torch.sum(z, dim=2)
        z = self.proj2(z)

        p = self.head(z).squeeze().permute(1, 0)
        return p, z
    
    def predict(
            self, 
            x: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            o, _ = self.forward(x)
            if kwargs["method"] == "mean":
                return o.sigmoid().mean(-1).flatten().cpu().numpy()
            elif kwargs["method"] == "max":
                return o.sigmoid().max(-1).values.flatten().cpu().numpy()

class CLIPformer(nn.Module):
    def __init__(
        self,
        backbone,
        device,
        n_layers: int,
        n_heads: int,
        mlp_dim: int,
        att_dim: int,
        num_classes: int = 1,
        cls_ration: int = 1,
        cls_dropout: float = 0.5,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.device = device

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone[0], device=device)
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hook to get the last layer tokens
        self.hook = Hook("transformer.resblocks.23.ln_2", self.clip.visual.transformer.resblocks[-1].ln_2)

        # Extension
        hidden_dim = backbone[1]
        self.encoder = Encoder(
            seq_length=CLIP_SEQ_LENGTH,
            num_layers=n_layers,
            num_heads=n_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

        # Patch Attention Pooling
        self.patch_attention_pool = PatchAttentionPool(
            att_dim=att_dim,
            n_heads=n_heads,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )

        # Classification head
        self.cls = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*cls_ration),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(hidden_dim*cls_ration, hidden_dim*cls_ration),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(hidden_dim*cls_ration, num_classes)
        )
        self.to(device)
        
    def forward(
            self, 
            x: torch.Tensor
    ):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = self.hook.output[1:, :, :]
        g = g.permute(1, 0, 2)
        g = self.encoder(g)
        g = self.patch_attention_pool(g)
        o = self.cls(g).squeeze(-1)
        return o, g

    def predict(
            self,
            x: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            o, _ = self.forward(x)
            return o.sigmoid().flatten().cpu().numpy()

class FlowModel(nn.Module):
    def __init__(
        self,
        backbone,
        flow,
        n_steps,
        n_proj,
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
        for i in range(n_proj):
            proj1_layers.extend(
                [
                    nn.Linear(backbone[1] if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers).to(device)
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))
        proj2_layers = [nn.Dropout()]
        for _ in range(n_proj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)
        
        # Initialize the trainable part of the model
        self.flow = MiniGlow(input_dim=proj_dim, num_steps=n_steps) if flow in "glow" else NormalizingFlow(input_dim=proj_dim, num_steps=n_steps)
        self.to(device)

    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]
        g = self.proj1(g.float())
        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)
        p = self.flow.log_prob(z)
        return p
    
    def predict(
            self, 
            x: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            return 1 - torch.exp(self.forward(x))
    
if __name__ == "__main__":
    # Example usage
    backbone = ("ViT-L/14", 1024)
    nproj = 2
    proj_dim = 512
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # model = FlowModel(backbone=backbone, flow="glow", n_steps=4, n_proj=2, proj_dim=512, device=device)
    model = CLIPformer(
        backbone=backbone,
        device=device,
        n_layers=4,
        n_heads=8,
        mlp_dim=1024,
        att_dim=512,
    )

    # Example input
    x = torch.randn(16, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(x)
    print("Output shape:", output.shape)