from functools import partial
from typing import Callable
import torch
import torch.nn as nn
import clip
import numpy as np
from src.nf import NormalizingFlow, MiniGlow
from src.utils import patchify_image
from open_clip import create_model_from_pretrained
from src.vision_transformer import Encoder
from torchvision.transforms.functional import five_crop
from einops import rearrange
import pickle
from typing import Union
import timm
from torchvision import transforms
from src.utils import custom_unfold

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

class GLIP(nn.Module):
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
        for param in self.clip.parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([257, 1, len(self.hooks), proj_dim])) # L_i x B x N x D_i
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
        self.proj1_cls = nn.Sequential(*proj1_layers)
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
        self.proj2_cls = nn.Sequential(*proj2_layers)
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
        self.head_cls = nn.Sequential(
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
            g = torch.stack([h.output for h in self.hooks], dim=2)
        
        g = self.proj1(g[1:, :, :, :].float())
        g_cls = self.proj1_cls(g[0, :, :, :].float())
        g = torch.cat([g_cls.unsqueeze(0), g], dim=0)

        z = torch.softmax(self.alpha, dim=2) * g
        z = torch.sum(z, dim=2)
        z_cls = z[0, :, :]
        z = z[1:, :, :]
        
        z = self.proj2(z)
        z_cls = self.proj2_cls(z_cls)
        
        p_cls = self.head_cls(z_cls).squeeze()
        p = self.head(z).squeeze()
        if p.dim() == 2:
            p = p.permute(1, 0)
        return p, z, p_cls, z_cls
    
    def forward_slide(self, img, stride=112, crop_size=224, batch_size_p=64, beta=0.5):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        imgs = custom_unfold(
            img,
            crop_size=crop_size,
            stride=stride,
        )
        logits = []
        for img in imgs:
            logits_img = []
            for i in range(0, img.shape[0], batch_size_p):
                batch_imgs = img[i:i + batch_size_p]
                logits_i, _, logits_o, _ = self.forward(batch_imgs)
                logits_img.append(logits_i.sigmoid().mean(-1) * beta + logits_o.sigmoid() * (1 - beta))
            logits.append(torch.cat(logits_img, dim=0).mean())
        return torch.stack(logits, dim=0)

    def predict(
        self, 
        x: Union[torch.Tensor, list[torch.Tensor]],
        **kwargs
    ):
        with torch.no_grad():
            beta = kwargs.get("beta", 1.0)
            #if kwargs.get("window_slide", False):
            o_l, _, o_g, _ = self.forward(x)
            return beta * o_l.sigmoid().mean(-1).flatten().cpu().numpy() + (1 - beta) * o_g.sigmoid().flatten().cpu().numpy()

class GL_RINE(nn.Module):
    def __init__(
        self,
        backbone,
        nproj,
        proj_dim,
        device,
    ):
        super().__init__()

        self.device = device
        self.proj_dim = proj_dim

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone[0], device=device)
        for param in self.clip.parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([256, 1, len(self.hooks), proj_dim])) # L_i x B x N x D_i
        self.alpha_cls = nn.Parameter(torch.randn([1, 1, len(self.hooks), proj_dim])) # L_i x B x N x D_i

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
        self.proj1_cls = nn.Sequential(*proj1_layers)

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
        self.proj2_cls = nn.Sequential(*proj2_layers)

        self.head = nn.Sequential(
            *[
                nn.Linear(2 * proj_dim, proj_dim), #TODO: try shared projection
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
        x_d = self.preprocess(x)
        z_cls = self.forward_global(x_d)
        _, z = self.forward_slide(x)
        z = z.mean(dim=1)   #TODO: try other pooling methods
        z = torch.cat([z_cls, z], dim=-1)
        p = self.head(z).squeeze()
        return p, z


    def forward_global(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]

        g = self.proj1_cls(g.float())
        z_cls = torch.softmax(self.alpha_cls, dim=2) * g
        z_cls = torch.sum(z_cls, dim=2)
        z_cls = self.proj2_cls(z_cls)

        # p = self.head(z_cls).squeeze()
        return z_cls

    def forward_window(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[1:, :, :, :]
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=2) * g
        z = torch.sum(z, dim=2)
        z = self.proj2(z)

        # p = self.head(z).squeeze()
        # if p.dim() == 2:
        #     p = p.permute(1, 0)
        return np.zeros((x.shape[0], 1)), z  #TODO: remove first part, just a placeholder


    def forward_slide(self, img, stride=112, crop_size=224, patch_size=14, reshape=True):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        assert stride % patch_size == 0, f"Stride muste be divisible by patch size ({patch_size})"
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        n_h, n_w = h_img // patch_size, w_img // patch_size
        s_h, s_w = h_stride // patch_size, w_stride // patch_size
        h_img, w_img = n_h * patch_size, n_w * patch_size
        h_w, w_w = h_crop // patch_size, w_crop // patch_size

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = img.new_zeros((batch_size, n_h, n_w))
        z = img.new_zeros((batch_size, n_h, n_w, self.proj_dim))
        count_mat = img.new_zeros((batch_size, n_h, n_w))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                h_1, w_1 = h_idx * s_h, w_idx * s_w
                h_2, w_2 = min(h_1 + h_w, n_h), min(w_1 + w_w, n_w)
                h_1, w_1 = max(h_2 - h_w, 0), max(w_2 - w_w, 0)

                crop_img = img[:, :, y1:y2, x1:x2]
                crop_img = self.preprocess(crop_img)
                crop_seg_logit, z = self.forward_window(crop_img)
                crop_seg_logit = z.reshape(-1, h_w, w_w)

                preds += nn.functional.pad(crop_seg_logit,
                               (int(w_1), int(preds.shape[2] - w_2), int(h_1),
                                int(preds.shape[1] - h_2)))
                z[:, h_1:h_2, w_1:w_2, :] += nn.functional.pad(z, (int(w_1), int(preds.shape[2] - w_2), int(h_1),
                                int(preds.shape[1] - h_2)))

                count_mat[:, h_1:h_2, w_1:w_2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        z = z / count_mat.unsqueeze(-1)

        if reshape:
            return preds.reshape(batch_size, -1), z.reshape(batch_size, -1, self.proj_dim)
        else:
            return preds, z

    def predict(
        self, 
        x: Union[torch.Tensor, list[torch.Tensor]],
        **kwargs
    ):
        with torch.no_grad():
            p = kwargs.get("p", 1)
            method = kwargs.get("method", "mean")
            if kwargs.get("window_slide", False):
                stride = kwargs.get("stride", 112)
                if isinstance(x, list):
                    o = []
                    for xi in x: 
                        o_i = self.forward_slide(xi, stride=stride)
                        if method == "mean":
                            o.append(o_i.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy())
                        elif method == "max":
                            o.append(o_i.sigmoid().max(-1).values.flatten().cpu().numpy())
                    return np.array(o).squeeze()
                else:
                    o = self.forward_slide(x, stride=stride)
                    if method == "mean":
                        return o.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy()
                    elif method == "max":
                        return o.sigmoid().max(-1).values.flatten().cpu().numpy()
            else:
                o, _ = self.forward(x)
                if method == "mean":
                    return o.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy()
                elif method == "max":
                    return o.sigmoid().max(-1).values.flatten().cpu().numpy()

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
        for param in self.clip.parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([256, 1, len(self.hooks), proj_dim])) # L_i x B x N x D_i
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
        for param in self.clip.parameters():
            param.requires_grad = True
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
    
    def forward_slide(self, img, stride=112, crop_size=224, patch_size=14, reshape=True):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        assert stride % patch_size == 0, f"Stride muste be divisible by patch size ({patch_size})"
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        n_h, n_w = h_img // patch_size, w_img // patch_size
        s_h, s_w = h_stride // patch_size, w_stride // patch_size
        h_img, w_img = n_h * patch_size, n_w * patch_size
        h_w, w_w = h_crop // patch_size, w_crop // patch_size

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = img.new_zeros((batch_size, n_h, n_w))
        count_mat = img.new_zeros((batch_size, n_h, n_w))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                h_1, w_1 = h_idx * s_h, w_idx * s_w
                h_2, w_2 = min(h_1 + h_w, n_h), min(w_1 + w_w, n_w)
                h_1, w_1 = max(h_2 - h_w, 0), max(w_2 - w_w, 0)

                crop_img = img[:, :, y1:y2, x1:x2]
                crop_img = transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                )(crop_img)
                crop_seg_logit, _ = self.forward(crop_img)
                crop_seg_logit = crop_seg_logit.reshape(-1, h_w, w_w)

                preds += nn.functional.pad(crop_seg_logit,
                               (int(w_1), int(preds.shape[2] - w_2), int(h_1),
                                int(preds.shape[1] - h_2)))

                count_mat[:, h_1:h_2, w_1:w_2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat

        if reshape:
            return preds.reshape(batch_size, -1)
        else:
            return preds
        
    def predict(
        self, 
        x: Union[torch.Tensor, list[torch.Tensor]],
        **kwargs
    ):
        with torch.no_grad():
            p = kwargs.get("p", 1)
            method = kwargs.get("method", "mean")
            if kwargs.get("window_slide", False):
                stride = kwargs.get("stride", 112)
                if isinstance(x, list):
                    o = []
                    for xi in x: 
                        o_i = self.forward_slide(xi, stride=stride)
                        if method == "mean":
                            o.append(o_i.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy())
                        elif method == "max":
                            o.append(o_i.sigmoid().max(-1).values.flatten().cpu().numpy())
                    return np.array(o).squeeze()
                else:
                    o = self.forward_slide(x, stride=stride)
                    if method == "mean":
                        return o.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy()
                    elif method == "max":
                        return o.sigmoid().max(-1).values.flatten().cpu().numpy()
            else:
                o, _ = self.forward(x)
                if method == "mean":
                    return o.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy()
                elif method == "max":
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
    
    def forward_slide(self, img, stride=128, crop_size=256, patch_size=16, reshape=True):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        assert stride % patch_size == 0, f"Stride muste be divisible by patch size ({patch_size})"
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        n_h, n_w = h_img // patch_size, w_img // patch_size
        s_h, s_w = h_stride // patch_size, w_stride // patch_size
        h_img, w_img = n_h * patch_size, n_w * patch_size
        h_w, w_w = h_crop // patch_size, w_crop // patch_size

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = img.new_zeros((batch_size, n_h, n_w))
        count_mat = img.new_zeros((batch_size, n_h, n_w))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                h_1, w_1 = h_idx * s_h, w_idx * s_w
                h_2, w_2 = min(h_1 + h_w, n_h), min(w_1 + w_w, n_w)
                h_1, w_1 = max(h_2 - h_w, 0), max(w_2 - w_w, 0)

                crop_img = img[:, :, y1:y2, x1:x2]
                crop_img = transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                )(crop_img)
                crop_seg_logit, _ = self.forward(crop_img)
                crop_seg_logit = crop_seg_logit.reshape(-1, h_w, w_w)

                preds += nn.functional.pad(crop_seg_logit,
                               (int(w_1), int(preds.shape[2] - w_2), int(h_1),
                                int(preds.shape[1] - h_2)))

                count_mat[:, h_1:h_2, w_1:w_2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat

        if reshape:
            return preds.reshape(batch_size, -1)
        else:
            return preds
    
    def predict(
            self, 
            x: Union[torch.Tensor, list[torch.Tensor]],
            **kwargs
    ):
        with torch.no_grad():
            if kwargs.get("window_slide", False):
                stride = kwargs.get("stride", 112)
                if isinstance(x, list):
                    o = []
                    for xi in x: 
                        o_i = self.forward_slide(xi, stride=stride)
                        if kwargs["method"] == "mean":
                            p = kwargs.get("p", 1)
                            o.append(o_i.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy())
                        elif kwargs["method"] == "max":
                            o.append(o_i.sigmoid().max(-1).values.flatten().cpu().numpy())
                    return np.array(o).squeeze()
                else:
                    o = self.forward_slide(x, stride=stride)
                    if kwargs["method"] == "mean":
                        p = kwargs.get("p", 1)
                        return o.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy()
                    elif kwargs["method"] == "max":
                        return o.sigmoid().max(-1).values.flatten().cpu().numpy()
                    else:
                        raise ValueError("Method not supported")
            else:
                o, _ = self.forward(x)
                if kwargs["method"] == "mean":
                    p = kwargs.get("p", 1)
                    return o.sigmoid().pow(p).mean(-1).pow(1/p).flatten().cpu().numpy()
                elif kwargs["method"] == "max":
                    return o.sigmoid().max(-1).values.flatten().cpu().numpy()
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


class RineModel(nn.Module):
    def __init__(self, backbone, nproj, proj_dim):
        super(RineModel, self).__init__()

        # Load and freeze CLIP
        self.clip, _ = clip.load(backbone[0], device="cpu")
        for _, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module) for name, module in self.clip.visual.named_modules() if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))

        proj1_layers = [
            nn.Dropout()
        ]

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
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]

        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)
        if p.dim() == 2:
            p = p.squeeze()
        return p, z

    def forward_slide(self, img, stride=112, crop_size=224, batch_size_p=64, beta=0.5):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        imgs = custom_unfold(
            img,
            crop_size=crop_size,
            stride=stride,
        )
        logits = []
        for img in imgs:
            for i in range(0, img.shape[0], batch_size_p):
                batch_imgs = img[i:i + batch_size_p]
                logits_i, _ = self.forward(batch_imgs)
                logits.append(logits_i)
        return torch.cat(logits, dim=0).sigmoid().mean(0).flatten()
    

    def predict(self, img, **kwargs):
        with torch.no_grad():
            if kwargs.get('window_slide', False):
                stride = kwargs.get('stride', 112)
                if isinstance(img, list):
                    o = []
                    for i in img:
                        o_i = self.forward_slide(i, stride=stride)
                        o.append(o_i.flatten().cpu().numpy())
                    return np.array(o).squeeze()
                else:
                    o = self.forward_slide(img, stride=stride)
                    return o.squeeze().flatten().cpu().numpy()
            else:
                logits, _ = self.forward(img)
                return logits.sigmoid().flatten().cpu().numpy()
        
    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        # for name in state_dict:
        #     exec(f'self.{name.replace(".", "[", 1).replace(".", "].", 1)} = torch.nn.Parameter(state_dict["{name}"])')
        self.load_state_dict(state_dict, strict=False)

class RINE_SigLIP(nn.Module):
    def __init__(
        self,
        backbone,
        nproj,
        proj_dim,
        device,
    ):
        super().__init__()

        self.device = device

        # Load and freeze SigLIP
        self.siglip, self.preprocess = create_model_from_pretrained(backbone[0], device=device)
        for name, param in self.siglip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.siglip.visual.named_modules()
            if "ls2" in name
        ]

        self.attention_pool = self.siglip.visual.attn_pool
        self.attention_pool.to(device)

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
            self.siglip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)
            g = self.attention_pool(g)
            print(g.shape)
        g = g.permute(1, 0, 2)
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z).squeeze()

        return p, z

# if __name__ == "__main__":
#     # Example usage
#     backbone = ("ViT-L/14", 1024)
#     nproj = 2
#     proj_dim = 512
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"

#     # model = FlowModel(backbone=backbone, flow="glow", n_steps=4, n_proj=2, proj_dim=512, device=device)
#     model = CLIPformer(
#         backbone=backbone,
#         device=device,
#         n_layers=4,
#         n_heads=8,
#         mlp_dim=1024,
#         att_dim=512,
#     )

#     # Example input
#     x = torch.randn(16, 3, 224, 224).to(device)
#     with torch.no_grad():
#         output = model(x)
#     print("Output shape:", output.shape)