# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from functools import partial
import pathlib
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch import optim as optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import five_crop
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.scheduler.cosine_lr import CosineLRScheduler
from einops import rearrange
from PIL import Image
from scipy import interpolate
import json

from src import vision_transformer_spai as vision_transformer
from src import filters

def patchify_image(
    img: torch.Tensor,
    patch_size: tuple[int, int],
    stride: tuple[int, int]
) -> torch.Tensor:
    """Splits an input image into patches,

    :param img: Input image of size (B, C, H, W).
    :param patch_size: (height, width) of patches.
    :param stride: Stride on (height, width) dimensions.

    :returns: Patchified image of size (B, L, C, patch_height, patch_width).
    """
    kh, kw = patch_size
    dh, dw = stride
    img = img.unfold(2, kh, dh).unfold(3, kw, dw)
    img = img.permute(0, 2, 3, 1, 4, 5)
    img = img.contiguous()
    img = img.view(img.size(0), -1, img.size(3), kh, kw)
    return img

def exportable_std(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard deviation operation exportable to ONNx.

    As of PyTorch 2.5 the dynamo exporter lacks support for the
    `torch.ops.prims.var.default` operation. Therefore, it fails
    to export variance operations applied to tensor.
    Github issue: https://github.com/pytorch/pytorch/issues/141969
    """
    m: torch.Tensor = x.mean(dim, keepdim=True)
    return  torch.sqrt(torch.pow(x-m, 2).sum(dim) / (x.size(dim) - 1)) # apply Bessel’s correction


# MAIN MODEL CLASS
class PatchBasedMFViT(nn.Module):
    def __init__(
        self,
        vit: vision_transformer.VisionTransformer,
        features_processor: 'FrequencyRestorationEstimator',
        cls_head: Optional[nn.Module],
        masking_radius: int,
        img_patch_size: int,
        img_patch_stride: int,
        cls_vector_dim: int,
        num_heads: int,
        attn_embed_dim: int,
        dropout: float = .0,
        frozen_backbone: bool = True,
        minimum_patches: int = 0,
        initialization_scope: str = "all"
    ) -> None:
        super().__init__()

        self.mfvit = MFViT(
            vit,
            features_processor,
            None,
            masking_radius,
            img_patch_size,
            frozen_backbone=frozen_backbone,
            initialization_scope=initialization_scope
        )

        self.img_patch_size: int = img_patch_size
        self.img_patch_stride: int = img_patch_stride
        self.minimum_patches: int = minimum_patches
        self.cls_vector_dim: int = cls_vector_dim

        # Cross-Attention with a learnable vector layers.
        dim_head: int = attn_embed_dim // num_heads
        self.heads = num_heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_kv = nn.Linear(cls_vector_dim, attn_embed_dim*2, bias=False)
        self.patch_aggregator = nn.Parameter(torch.zeros((num_heads, 1, attn_embed_dim//num_heads)))
        nn.init.trunc_normal_(self.patch_aggregator, std=.02)
        self.to_out = nn.Sequential(
            nn.Linear(attn_embed_dim, cls_vector_dim, bias=False),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(cls_vector_dim)
        self.cls_head = cls_head

        if initialization_scope == "all":
            self.apply(_init_weights)
        elif initialization_scope == "local":
            # Initialize only the newly added components, by excluding mfvit.
            for m_name, m in self._modules.items():
                if m_name == "mfvit":
                    continue
                else:
                    m.apply(_init_weights)
        else:
            raise TypeError(f"Non-supported weight initialization type: {initialization_scope}")

    def forward(
        self,
        x: Union[torch.Tensor, list[torch.Tensor]],
        feature_extraction_batch_size: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list['AttentionMask']]]:
        """Forward pass of a batch of images.

        The images should not have been normalized before and the value of each pixel should
        lie in [0, 1].

        :param x: B x C x H x W
        :param feature_extraction_batch_size:
        :param export_dirs:
        """
        if isinstance(x, torch.Tensor):
            x =  self.forward_batch(x)
        elif isinstance(x, list):
            if feature_extraction_batch_size is None:
                feature_extraction_batch_size = len(x)
            else:
                x = self.forward_arbitrary_resolution_batch(x, feature_extraction_batch_size)
        else:
            raise TypeError('x must be a tensor or a list of tensors')

        return x
    
    def predict(
            self,
            x: Union[torch.Tensor, list[torch.Tensor]],
        ) -> torch.Tensor:
        """Predicts the class of an image.

        :param x: B x C x H x W
        """
        with torch.no_grad():
            x = self.forward(x)
        return x.sigmoid().flatten().cpu().numpy()

    def patches_attention(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        drop_bottom: bool = False,
        drop_top: bool = False
    ) -> torch.Tensor:
        """Perform cross attention between a learnable vector and the patches of an image."""
        aggregator: torch.Tensor = self.patch_aggregator.expand(x.size(0), -1, -1, -1)
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        dots = torch.matmul(aggregator, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x)
        x = x.squeeze(dim=1)
        if return_attn:
            return x, attn
        else:
            return x

    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        x = patchify_image(
            x,
            (self.img_patch_size, self.img_patch_size),
            (self.img_patch_stride, self.img_patch_stride)
        )  # B x L x C x H x W

        patch_features: list[torch.Tensor] = []
        for i in range(x.size(1)):
            patch_features.append(self.mfvit(x[:, i]))
        x = torch.stack(patch_features, dim=1)  # B x L x D
        del patch_features

        x = self.patches_attention(x)  # B x D
        x = self.norm(x)  # B x D
        x = self.cls_head(x)  # B x 1

        return x

    def forward_arbitrary_resolution_batch(
        self,
        x: list[torch.Tensor],
        feature_extraction_batch_size: int
    ) -> torch.Tensor:
        """Forward pass of a batch of images of different resolutions.

        Batch size on the tensors should equal one.

        :param x: list of 1 x C x H_i x W_i tensors, where i denote the i-th image in the list.
        :param feature_extraction_batch_size:

        :returns: A B x 1 tensor.
        """
        # Rearrange the patches from all images into a single tensor.
        patched_images: list[torch.Tensor] = []
        for img in x:
            patched: torch.Tensor = patchify_image(
                img,
                (self.img_patch_size, self.img_patch_size),
                (self.img_patch_stride, self.img_patch_stride)
            )  # 1 x L_i x C x H x W
            if patched.size(1) < self.minimum_patches:
                patched: tuple[torch.Tensor, ...] = five_crop(
                    img, [self.img_patch_size, self.img_patch_size]
                )
                patched: torch.Tensor = torch.stack(patched, dim=1)
            patched_images.append(patched)
        x = patched_images
        del patched_images
        # x = [
        #     patchify_image(
        #         img,
        #         (self.img_patch_size, self.img_patch_size),
        #         (self.img_patch_stride, self.img_patch_stride)
        #     )  # 1 x L_i x C x H x W
        #     for img in x
        # ]
        img_patches_num: list[int] = [img.size(1) for img in x]
        x = torch.cat(x, dim=1)  # 1 x SUM(L_i) x C x H x W
        x = x.squeeze(dim=0)  # SUM(L_i) x C x H x W

        # Process the patches in groups of feature_extraction_batch_size.
        features: list[torch.Tensor] = []
        for i in range(0, x.size(0), feature_extraction_batch_size):
            features.append(self.mfvit(x[i:i+feature_extraction_batch_size]))
        x = torch.cat(features, dim=0)  # SUM(L_i) x D
        del features

        # Attend to patches according to the image they belong to.
        attended: list[torch.Tensor] = []
        processed_sum: int = 0
        for i in img_patches_num:
            attended.append(self.patches_attention(x[processed_sum:processed_sum+i].unsqueeze(0)))
            processed_sum += i
        x = torch.cat(attended, dim=0)  # B x D
        del attended

        x = self.norm(x)  # B x D
        x = self.cls_head(x)  # B x 1

        return x

    def get_vision_transformer(self) -> vision_transformer.VisionTransformer:
        return self.mfvit.get_vision_transformer()

    def unfreeze_backbone(self) -> None:
        self.mfvit.unfreeze_backbone()

    def freeze_backbone(self) -> None:
        self.mfvit.freeze_backbone()

    def load_weights(self, ckpt: str) -> None:
        """Load weights from a checkpoint.

        :param ckpt: Path to the checkpoint file.
        """
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
        ckpt_model = checkpoint['model']
        self.load_state_dict(ckpt_model, strict=False)


class MFViT(nn.Module):
    """Model that constructs features according to the ability to restore missing frequencies."""
    def __init__(
        self,
        vit: vision_transformer.VisionTransformer,
        features_processor: 'FrequencyRestorationEstimator',
        cls_head: Optional[nn.Module],
        masking_radius: int,
        img_size: int,
        frozen_backbone: bool = True,
        initialization_scope: str = "all"
    ):
        super().__init__()
        self.vit = vit
        self.features_processor = features_processor
        self.cls_head = cls_head

        if initialization_scope == "all":
            self.apply(_init_weights)
        elif initialization_scope == "local":
            # Initialize only the newly added components, by excluding vit.
            for m_name, m in self._modules.items():
                if m_name == "vit":
                    continue
                else:
                    m.apply(_init_weights)
        else:
            raise TypeError(f"Non-supported weight initialization type: {initialization_scope}")

        self.frozen_backbone: bool = frozen_backbone

        self.frequencies_mask: nn.Parameter = nn.Parameter(
            filters.generate_circular_mask(img_size, masking_radius),
            requires_grad=False
        )

        if (isinstance(self.vit, vision_transformer.VisionTransformer)):
            # ImageNet normalization
            self.backbone_norm = transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            )
        else:
            raise TypeError(f"Unsupported backbone type: {type(vit)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a batch of images.

        The images should not have been normalized before and the value of each pixel should
        lie in [0, 1].

        :param x: B x C x H x W
        """

        low_freq: torch.Tensor
        hi_freq: torch.Tensor
        low_freq, hi_freq = filters.filter_image_frequencies(x.float(), self.frequencies_mask)

        low_freq = torch.clamp(low_freq, min=0., max=1.).to(x.dtype)
        hi_freq = torch.clamp(hi_freq, min=0., max=1.).to(x.dtype)

        # Normalize all components according to ImageNet.
        x = self.backbone_norm(x)
        low_freq = self.backbone_norm(low_freq)
        hi_freq = self.backbone_norm(hi_freq)

        if self.frozen_backbone:
            with torch.no_grad():
                x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)
        else:
            x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)

        x = self.features_processor(x, low_freq, hi_freq)
        if self.cls_head is not None:
            x = self.cls_head(x)

        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.forward(x)
            return x.sigmoid().flatten().cpu().numpy()

    def forward_with_export(self, x: torch.Tensor, export_file: pathlib.Path) -> torch.Tensor:
        """Forward pass of a batch of images.

        The images should not have been normalized before and the value of each pixel should
        lie in [0, 1].

        :param x: B x C x H x W
        :export_file:
        """

        low_freq: torch.Tensor
        hi_freq: torch.Tensor
        low_freq, hi_freq = filters.filter_image_frequencies(x.float(), self.frequencies_mask)

        low_freq = torch.clamp(low_freq, min=0., max=1.).to(x.dtype)
        hi_freq = torch.clamp(hi_freq, min=0., max=1.).to(x.dtype)

        export_file.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray((x.detach().cpu().permute(0, 2, 3, 1).squeeze(dim=0).numpy()*255).astype(np.uint8)).save(export_file)
        Image.fromarray((hi_freq.detach().cpu().permute(0, 2, 3, 1).squeeze(dim=0).numpy()*255).astype(np.uint8)).save(
            f"{export_file.parent}/{export_file.stem}_hi_freq{export_file.suffix}")
        Image.fromarray((low_freq.detach().cpu().permute(0, 2, 3, 1).squeeze(dim=0).numpy() * 255).astype(np.uint8)).save(
            f"{export_file.parent}/{export_file.stem}_low_freq{export_file.suffix}")

        # Normalize all components according to ImageNet.
        x = self.backbone_norm(x)
        low_freq = self.backbone_norm(low_freq)
        hi_freq = self.backbone_norm(hi_freq)

        if self.frozen_backbone:
            with torch.no_grad():
                x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)
        else:
            x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)

        x = self.features_processor(x, low_freq, hi_freq)
        if self.cls_head is not None:
            x = self.cls_head(x)

        return x

    def get_vision_transformer(self) -> vision_transformer.VisionTransformer:
        return self.vit

    def unfreeze_backbone(self) -> None:
        self.frozen_backbone = False

    def freeze_backbone(self) -> None:
        self.frozen_backbone = True

    def _extract_features(
        self,
        x: torch.Tensor,
        low_freq: torch.Tensor,
        hi_freq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.vit(x)
        low_freq = self.vit(low_freq)
        hi_freq = self.vit(hi_freq)
        return x, low_freq, hi_freq
    
    def load_weights(self, ckpt: str) -> None:
        """Load weights from a checkpoint.

        :param ckpt: Path to the checkpoint file.
        """
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
        ckpt_model = checkpoint['model']
        self.load_state_dict(ckpt_model, strict=False)


class FrequencyRestorationEstimator(nn.Module):

    def __init__(
        self,
        features_num: int,
        input_dim: int,
        proj_dim: int,
        proj_layers: int,
        patch_projection: bool = False,
        patch_projection_per_feature: bool = False,
        proj_last_layer_activation_type: Optional[str] = "gelu",
        original_image_features_branch: bool = False,
        dropout: float = 0.5,
        disable_reconstruction_similarity: bool = False
    ):
        super().__init__()

        if proj_last_layer_activation_type == "gelu":
            proj_last_layer_activation = nn.GELU
        elif proj_last_layer_activation_type is None:
            proj_last_layer_activation = nn.Identity
        else:
            raise RuntimeError("Unsupported activation type for the "
                               f"last projection layer: {proj_last_layer_activation_type}")

        if patch_projection and patch_projection_per_feature:
            self.patch_projector: nn.Module = FeatureSpecificProjector(
                features_num, proj_layers, input_dim, proj_dim, proj_last_layer_activation,
                dropout=dropout
            )
        elif patch_projection:
            self.patch_projector: nn.Module = Projector(
                proj_layers, input_dim, proj_dim, proj_last_layer_activation, dropout=dropout
            )
        else:
            self.patch_projector: nn.Module = nn.Identity()

        self.original_features_processor = None
        if original_image_features_branch:
            self.original_features_processor = FeatureImportanceProjector(
                features_num, proj_dim, proj_dim, proj_layers, dropout=dropout
            )

        # A flag that when set stops the computation of reconstruction similarity scores.
        # Useful for performing ablation studies.
        self.disable_reconstruction_similarity: bool = disable_reconstruction_similarity
        if self.disable_reconstruction_similarity:
            assert self.original_features_processor is not None, \
                ("Frequency Reconstruction Similarity cannot be disabled without "
                 "Original Features Processor.")

    def forward(
        self,
        x: torch.Tensor,
        low_freq: torch.Tensor,
        hi_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: Dimensionality B x N x L x D where N the number of intermediate layers.
        :param low_freq:
        :param hi_freq:

        :returns: Dimensionality B x (6 * N)
        """
        orig = self.patch_projector(x)  # B x N x L x D
        low_freq = self.patch_projector(low_freq)  # B x N x L x D
        hi_freq = self.patch_projector(hi_freq)  # B x N x L x D

        if self.disable_reconstruction_similarity:
            x = self.original_features_processor(orig)  # B x proj_dim
        else:
            sim_x_low_freq: torch.Tensor = F.cosine_similarity(orig, low_freq, dim=-1)  # B x N x L
            sim_x_hi_freq: torch.Tensor = F.cosine_similarity(orig, hi_freq, dim=-1)  # B x N x L
            sim_low_freq_hi_freq: torch.Tensor = F.cosine_similarity(low_freq, hi_freq, dim=-1)  # B x N x L

            sim_x_low_freq_mean: torch.Tensor = sim_x_low_freq.mean(dim=-1)  # B x N
            sim_x_low_freq_std: torch.Tensor = sim_x_low_freq.std(dim=-1)  # B x N
            sim_x_hi_freq_mean: torch.Tensor = sim_x_hi_freq.mean(dim=-1)  # B x N
            sim_x_hi_freq_std: torch.Tensor = sim_x_hi_freq.std(dim=-1)  # B x N
            sim_low_freq_hi_freq_mean: torch.Tensor = sim_low_freq_hi_freq.mean(dim=-1)  # B x N
            sim_low_freq_hi_freq_std: torch.Tensor = sim_low_freq_hi_freq.std(dim=-1)  # B x N

            x: torch.Tensor = torch.cat([
                sim_x_low_freq_mean,
                sim_x_low_freq_std,
                sim_x_hi_freq_mean,
                sim_x_hi_freq_std,
                sim_low_freq_hi_freq_mean,
                sim_low_freq_hi_freq_std
            ], dim=1)  # B x (6 * N)

            if self.original_features_processor is not None:
                orig = self.original_features_processor(orig)  # B x proj_dim
                x = torch.cat([x, orig], dim=1)  # B x (proj_dim + 6 * N)

        return x

    def exportable_forward(
        self,
        x: torch.Tensor,
        low_freq: torch.Tensor,
        hi_freq: torch.Tensor
    ) -> torch.Tensor:
        orig = self.patch_projector(x)  # B x N x L x D
        low_freq = self.patch_projector(low_freq)  # B x N x L x D
        hi_freq = self.patch_projector(hi_freq)  # B x N x L x D

        sim_x_low_freq: torch.Tensor = F.cosine_similarity(orig, low_freq, dim=-1)  # B x N x L
        sim_x_hi_freq: torch.Tensor = F.cosine_similarity(orig, hi_freq, dim=-1)  # B x N x L
        sim_low_freq_hi_freq: torch.Tensor = F.cosine_similarity(low_freq, hi_freq,
                                                                 dim=-1)  # B x N x L

        sim_x_low_freq_mean: torch.Tensor = sim_x_low_freq.mean(dim=-1)  # B x N
        sim_x_low_freq_std: torch.Tensor = exportable_std(sim_x_low_freq, dim=-1)  # B x N
        sim_x_hi_freq_mean: torch.Tensor = sim_x_hi_freq.mean(dim=-1)  # B x N
        sim_x_hi_freq_std: torch.Tensor = exportable_std(sim_x_hi_freq, dim=-1)  # B x N
        sim_low_freq_hi_freq_mean: torch.Tensor = sim_low_freq_hi_freq.mean(dim=-1)  # B x N
        sim_low_freq_hi_freq_std: torch.Tensor = exportable_std(
            sim_low_freq_hi_freq, dim=-1
        )  # B x N

        x: torch.Tensor = torch.cat([
            sim_x_low_freq_mean,
            sim_x_low_freq_std,
            sim_x_hi_freq_mean,
            sim_x_hi_freq_std,
            sim_low_freq_hi_freq_mean,
            sim_low_freq_hi_freq_std
        ], dim=1)  # B x (6 * N)

        orig = self.original_features_processor.exportable_forward(orig)  # B x proj_dim
        x = torch.cat([x, orig], dim=1)  # B x (proj_dim + 6 * N)

        return x


class FeatureSpecificProjector(nn.Module):
    def __init__(
            self,
            intermediate_features_num: int,
            proj_layers: int,
            input_dim: int,
            proj_dim: int,
            last_layer_activation = nn.GELU,
            dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleList([
            Projector(proj_layers, input_dim, proj_dim, last_layer_activation, dropout=dropout)
            for _ in range(intermediate_features_num)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected: list[torch.Tensor] = []
        for i, projector in enumerate(self.projectors):
            projected.append(projector(x[:, i, :, :]))
        x = torch.stack(projected, dim=1)
        return x


class Projector(nn.Module):
    def __init__(
        self,
        proj_layers: int,
        input_dim: int,
        proj_dim: int,
        last_layer_activation = nn.GELU,
        input_norm: bool = True,
        output_norm: bool = True,
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(input_dim) if input_norm else nn.Identity()
        patch_proj_layers: list[nn.Module] = [nn.Dropout(dropout)]
        for i in range(proj_layers):
            patch_proj_layers.extend(
                [
                    nn.Linear(input_dim if i == 0 else proj_dim, proj_dim),
                    nn.GELU() if i < proj_layers - 1 else last_layer_activation(),
                    nn.Dropout(dropout),
                ]
            )
        self.projector: nn.Sequential = nn.Sequential(*patch_proj_layers)
        self.norm2 = nn.LayerNorm(proj_dim) if output_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.projector(x)
        x = self.norm2(x)
        return x


class ClassificationHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        mlp_ratio: int = 1,
        dropout: float = 0.5
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim*mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*mlp_ratio, input_dim*mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*mlp_ratio, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
         return self.head(x)


class FeatureImportanceProjector(nn.Module):

    def __init__(
        self,
        intermediate_features_num: int,
        input_dim: int,
        proj_dim: int,
        proj_layers: int,
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.randn([1, intermediate_features_num, proj_dim]))
        self.proj1 = Projector(proj_layers, 2*proj_dim, proj_dim, input_norm=False, dropout=dropout)
        self.proj2 = Projector(proj_layers, proj_dim, proj_dim, input_norm=False, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean: torch.Tensor = x.mean(dim=2)  # B x N x input_dim
        x_std: torch.Tensor = x.std(dim=2)  # B x N x input_dim
        x = torch.cat([x_mean, x_std], dim=-1)  # B x N x 2*input_dim

        x = self.proj1(x)  # B x N x 2*proj_dim
        x = torch.softmax(self.alpha, dim=1) * x
        x = torch.sum(x, dim=1)  # B x proj_dim
        x = self.proj2(x)

        return x

    def exportable_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean: torch.Tensor = x.mean(dim=2)  # B x N x input_dim
        x_std: torch.Tensor = exportable_std(x, dim=2)  # B x N x input_dim
        x = torch.cat([x_mean, x_std], dim=-1)  # B x N x 2*input_dim

        x = self.proj1(x)  # B x N x 2*proj_dim
        x = torch.softmax(self.alpha, dim=1) * x
        x = torch.sum(x, dim=1)  # B x proj_dim
        x = self.proj2(x)

        return x

@dataclasses.dataclass
class AttentionMask:
    mask: Optional[pathlib.Path] = None
    overlay: Optional[pathlib.Path] = None
    overlayed_image: Optional[pathlib.Path] = None


def build_mf_vit(config) -> MFViT:
    # Build features extractor.
    vit: vision_transformer.VisionTransformer = vision_transformer.build_vit(config)
    initialization_scope: str = "all"

    # Build features processor.
    fre: FrequencyRestorationEstimator = FrequencyRestorationEstimator(
        features_num=len(config.MODEL.VIT.INTERMEDIATE_LAYERS),
        input_dim=config.MODEL.VIT.EMBED_DIM,
        proj_dim=config.MODEL.VIT.PROJECTION_DIM,
        proj_layers=config.MODEL.VIT.PROJECTION_LAYERS,
        patch_projection=config.MODEL.VIT.PATCH_PROJECTION,
        patch_projection_per_feature=config.MODEL.VIT.PATCH_PROJECTION_PER_FEATURE,
        proj_last_layer_activation_type=config.MODEL.FRE.PROJECTOR_LAST_LAYER_ACTIVATION_TYPE,
        original_image_features_branch=config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH,
        dropout=config.MODEL.SID_DROPOUT,
    )
    cls_vector_dim: int = 6 * len(config.MODEL.VIT.INTERMEDIATE_LAYERS)
    if config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH:
        cls_vector_dim += config.MODEL.VIT.PROJECTION_DIM

    cls_head: Optional[ClassificationHead]
    if config.TRAIN.MODE == "contrastive":
        cls_head = None
    elif config.TRAIN.MODE == "supervised":
        # Build classification head.
        cls_head = ClassificationHead(
            input_dim=cls_vector_dim,
            num_classes=config.MODEL.NUM_CLASSES if config.MODEL.NUM_CLASSES > 2 else 1,
            mlp_ratio=config.MODEL.CLS_HEAD.MLP_RATIO,
            dropout=config.MODEL.SID_DROPOUT
        )
    else:
        raise RuntimeError(f"Unsupported train mode: {config.TRAIN.MODE}")

    if config.MODEL.RESOLUTION_MODE == "fixed":
        model = MFViT(
            vit,
            fre,
            cls_head,
            masking_radius=config.MODEL.FRE.MASKING_RADIUS,
            img_size=config.DATA.IMG_SIZE
        )
    elif config.MODEL.RESOLUTION_MODE == "arbitrary":
        model = PatchBasedMFViT(
            vit,
            fre,
            cls_head,
            masking_radius=config.MODEL.FRE.MASKING_RADIUS,
            img_patch_size=config.DATA.IMG_SIZE,
            img_patch_stride=config.MODEL.PATCH_VIT.PATCH_STRIDE,
            cls_vector_dim=cls_vector_dim,
            attn_embed_dim=config.MODEL.PATCH_VIT.ATTN_EMBED_DIM,
            num_heads=config.MODEL.PATCH_VIT.NUM_HEADS,
            dropout=config.MODEL.SID_DROPOUT,
            minimum_patches=config.MODEL.PATCH_VIT.MINIMUM_PATCHES,
            initialization_scope=initialization_scope
        )
    else:
        raise RuntimeError(f"Unsupported resolution mode: {config.MODEL.RESOLUTION_MODE}")

    return model


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(m.weight, std=.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def remap_pretrained_keys_vit(model, checkpoint_model):
    # Duplicate shared rel_pos_bias to each layer
    if getattr(model, 'use_rel_pos_bias', False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        num_layers = model.get_num_layers()
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias
    
    return checkpoint_model

def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1
    
def get_vit_layer(name, num_layers):
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    else:
        return num_layers - 1
    
def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def build_finetune_optimizer(config, model):
    print('>>>>>>>>>> Build Optimizer for Fine-tuning Stage')
    if config.MODEL.TYPE == 'swin':
        depths = config.MODEL.SWIN.DEPTHS
        num_layers = sum(depths)
        get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
    elif config.MODEL.TYPE == 'vit':
        num_layers = config.MODEL.VIT.DEPTH
        get_layer_func = partial(get_vit_layer, num_layers=num_layers + 2)
    else:
        raise NotImplementedError
    
    scales = list(config.TRAIN.LAYER_DECAY ** i for i in reversed(range(num_layers + 2)))
    
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        print(f'No weight decay: {skip}')
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
        print(f'No weight decay keywords: {skip_keywords}')

    parameters = get_finetune_param_groups(
        model, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY,
        get_layer_func, scales, skip, skip_keywords)
    
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    print(optimizer)
    return optimizer

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config.TRAIN.MIN_LR,
        warmup_lr_init=config.TRAIN.WARMUP_LR,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return lr_scheduler

def load_pretrained(
    config,
    model,
    checkpoint_path: Optional[pathlib.Path] = None,
) -> int:
    if checkpoint_path is None:
        checkpoint_path = pathlib.Path(config.PRETRAINED)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint['model']
    checkpoint_epoch: Optional[int] = checkpoint.get('epoch', None)

    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {
            k.replace('encoder.', ''): v
            for k, v in checkpoint_model.items() if k.startswith('encoder.')
        }
    else:
        print('Detect non-pre-trained model, pass without doing anything.')

    if config.MODEL.TYPE == 'vit':
        checkpoint = remap_pretrained_keys_vit(model, checkpoint_model)
    else:
        raise NotImplementedError

    msg = model.load_state_dict(checkpoint_model, strict=False)
    
    del checkpoint
    torch.cuda.empty_cache()
    print(f">>>>>>>>>> loaded successfully '{config.PRETRAINED}'")

    return checkpoint_epoch