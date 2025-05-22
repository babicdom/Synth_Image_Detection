import os
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from src.utils import patchify_image, get_transform
from PIL import Image
import pickle
from src.models import IntermediatePatch

def visualize_patchify(image, patch_size=(224, 224), stride=(16, 16)):
    """
    Visualize the patchified image.
    """
    patches = patchify_image(image, patch_size, stride)
    
    num_patches = patches.shape[1]
    print(f"Original image shape: {image.shape}")
    print(f"Patchified image shape: {patches.shape}")
    num_patches_h = int(np.sqrt(num_patches))
    num_patches_w = int(np.ceil(num_patches / num_patches_h))
    print(f"Number of patches: {num_patches}")    
    fig, axes = plt.subplots(
        nrows=num_patches_h,
        ncols=num_patches_w,
        figsize=(num_patches_w * 2, num_patches_h * 2),
    )
    if num_patches == 1:
        axes = np.array([[axes]])
    for i in range(num_patches):
        ax = axes[i // num_patches_w, i % num_patches_w]
        im = transforms.ToPILImage()(patches[0, i])
        ax.imshow(im)
        ax.set_title(f"Patch {i + 1}")
        ax.axis('off')
    plt.tight_layout()
    os.makedirs("results/visualizations", exist_ok=True)
    plt.savefig(os.path.join("results", "visualizations", "patchified_image.png"))

def visualize_image_impact(
        img: torch.Tensor,
        model,
        **kwargs
    ) -> None:
    """
    Visualizes the impact of each pixel on the model's output.

    :param img: Input image of size (C, H, W).
    :param model: Model to be used for prediction.
    """
    # Ensure the image is in the correct format
    if img.dim() == 3:
        img = img.unsqueeze(0)  # Add batch dimension
    img.requires_grad = True

    output = model.forward_with_grad(img)[0].mean()
    output.backward()
    gradients = img.grad

    gradients = gradients.view(gradients.size(0), -1, img.size(2), img.size(3))
    gradients = gradients.permute(0, 2, 3, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.axis("off")
    plt.title("Input Image")
    plt.subplot(1, 2, 2)
    plt.imshow(gradients[0].detach().cpu().numpy(), cmap="hot")
    plt.axis("off")
    plt.title("Gradient Impact")
    plt.savefig(f"gradients.png")

def visualize_patches(
        img: torch.Tensor,
        model,
        patch_size: tuple[int, int],
        stride: tuple[int, int],
        **kwargs
    ) -> None:
    """
    Visualizes the patches of an image.

    :param img: Input image of size (C, H, W).
    :param model: Model to be used for prediction.
    """
    # Ensure the image is in the correct format
    if img.dim() == 3:
        img = img.unsqueeze(0)  # Add batch dimension
    img = patchify_image(img, patch_size, stride)
    
    plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.axis("off")
    plt.title("Patches")
    plt.savefig(f"patches.png")
    plt.show()

if __name__ == "__main__":
    # Load a sample image
    image_path = os.path.join("data", "test", "synthbuster", "dalle3", "1_fake", "r09a79474t.png")
    image = Image.open(image_path).convert("RGB")
    image = transforms.ToTensor()(image)  # Convert to tensor
    image = image.unsqueeze(0)  # Add batch dimension
    visualize_patchify(image, patch_size=(224, 224), stride=(16, 16))
    print("Patchified image saved to results/visualizations/patchified_image.png")

    # img = Image.open("data/test/diffusion_datasets/dalle/1_fake/efgchmasis.png")
    # img = img.convert("RGB")
    # tr = get_transform("val")

    # experiment = pickle.load(
    #     open(f"ckpt/IntermediatePatch/3_nproj_512_proj_dim/experiment.pickle", "rb")
    # )
    # model = IntermediatePatch(
    #     backbone=experiment["backbone"],
    #     nproj=experiment["nproj"],
    #     proj_dim=experiment["proj_dim"],
    #     device=torch.device("cuda:0"),
    # )
    # model.load_state_dict(
    #     torch.load(f"ckpt/IntermediatePatch/3_nproj_512_proj_dim/train.pth", map_location="cuda:0")
    # )

    # visualize_image_impact(
    #     img=tr(img).unsqueeze(0).to("cuda:0"),
    #     model=model,
    # )