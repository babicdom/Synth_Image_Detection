import os
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.utils import patchify_image, get_transform, get_loader, image_enlisting_collate_fn, get_generators, get_real_images, custom_unfold
from src.data import EvaluationDataset, TestDataset
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pickle
from src.models import IntermediatePatch, SigLIPIntermediate, RineModel
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, auc
import tqdm
import json
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import floor
import random

def plot_pr(labels, output, ax, name=""):
    pr_curve = precision_recall_curve(labels, output)
    ap = average_precision_score(labels, output)
    
    precision_recall_display = PrecisionRecallDisplay(
        precision=pr_curve[0],
        recall=pr_curve[1],
        average_precision=ap*100,
    )
    precision_recall_display.plot(ax=ax)
    ax.set_title(f"PR Curve {name} (AP = {ap*100:.2f})")

def plot_auc_roc(labels, output, ax, name=""):
    auc_roc_curve = roc_curve(labels, output)
    auc_roc = roc_auc_score(labels, output)
    
    roc_curve_display = RocCurveDisplay(
        fpr=auc_roc_curve[0],
        tpr=auc_roc_curve[1],
        roc_auc=auc_roc*100,
        estimator_name=name,
    )
    roc_curve_display.plot(ax=ax)
    ax.set_title(f"ROC Curve {name} (AUC = {auc_roc*100:.2f})")

def visualize_detection_curves(model_names, base_path="results/curves"):
    """Visualizes detection performance curves with proper sklearn-style formatting.
    
    Args:
        model_names (list): List of model names to visualize
        base_path (str): Path containing the curve JSON files
    """
    os.makedirs(f"results/visualizations/comparison", exist_ok=True)
    plt.style.use('seaborn-v0_8')
    suffixes = ['all', 'ldm', 'gan']

    for row_idx, suffix in enumerate(suffixes):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        for idx, model in enumerate(model_names):
            file_path = os.path.join(base_path, f"{model}_{suffix}_curves.json")
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r') as f:
                curve_data = json.load(f)[0]

            # Highlight first model, dim others
            if idx == 0:
                lw, alpha, zorder = 3, 1.0, 3
            else:
                lw, alpha, zorder = 1.5, 0.3, 1
            if idx == 0:
                model = "Ours"
            elif model == "Rine_latent_diffusion":
                model = "Rine"
            elif model == "Rine_all_classes":
                model = "Rine"
            elif model == "SPAI_progan":
                model = "SPAI"

            # ROC Curve
            roc = curve_data['roc_curve']
            roc_auc = auc(roc[0], roc[1])
            axs[0].plot(
                roc[0], roc[1],
                lw=lw, alpha=alpha, zorder=zorder,
                label=f"{model} (AUC = {100*roc_auc:.2f})"
            )

            # Precision-Recall Curve
            precision, recall, _ = curve_data['precision_recall_curve']
            average_precision = auc(recall, precision)
            axs[1].plot(
                recall, precision,
                lw=lw, alpha=alpha, zorder=zorder,
                label=f"{model} (AP = {100*average_precision:.2f})"
            )

        # Format ROC subplot
        axs[0].set_title(f"ROC Curve Comparison", fontsize=14)
        axs[0].set_xlabel('False Positive Rate', fontsize=12)
        axs[0].set_ylabel('True Positive Rate', fontsize=12)
        axs[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axs[0].grid(True)
        axs[0].legend(loc='lower right', fontsize=10)

        # Format Precision-Recall subplot
        axs[1].set_title(f"Precision-Recall Curve Comparison", fontsize=14)
        axs[1].set_xlabel('Recall', fontsize=12)
        axs[1].set_ylabel('Precision', fontsize=12)
        axs[1].set_ylim([0.0, 1.05])
        axs[1].grid(True)
        axs[1].legend(loc='lower left', fontsize=10)
            
        # Save the figure
        plt.tight_layout(pad=3.0)
        plt.savefig(f"results/visualizations/comparison/{suffix}_comparison.png")

def visualize_prediction_distribution(
        data,
        model,
        device: torch.device,
        model_name="IntermediatePatch",
    ):
    """
    Visualizes the prediction distribution of the model.
    """
    model.eval()

    for generator, dl in data:
        generator = generator.split("/")[-1]
        real_mean_pred = []
        real_max_pred = []
        fake_mean_pred = []
        fake_max_pred = []
        real_pred_g_mean_3 = []
        fake_pred_g_mean_3 = []
        real_pred_g_mean_15 = []
        fake_pred_g_mean_15 = []
        max_pred_reverse = []
        g_mean_pred_reverse = []
        reals = []
        fakes = []
        labels_ = []
        output_mean =[]
        output_max = []
        output_g_mean_3 = []

        print(f'Fake: {len(dl.dataset.fake)}, Real: {len(dl.dataset.real)}, Total: {len(dl.dataset.images)}')
        with torch.no_grad():
            for data in tqdm.tqdm(dl, desc=f"Extracting tokens for {generator}", unit="batch"):
                images, labels, _ = data
                images, labels = images.float().to(device), labels.to(device)
                output, _ = model(images)
                output = output.sigmoid()

                output_mean.extend(output.mean(-1).flatten().cpu().numpy())
                output_max.extend(output.max(-1).values.flatten().cpu().numpy())
                output_g_mean_3.extend(output.pow(3).mean(-1).pow(1/3).flatten().cpu().numpy())
                labels_.extend(labels.cpu().numpy())

                output_fake = output[labels == 1]
                output_real = output[labels == 0]
                real_mean_pred.extend(output_real.mean(-1).flatten().cpu().numpy())
                real_max_pred.extend(output_real.max(-1).values.flatten().cpu().numpy())
                fake_mean_pred.extend(output_fake.mean(-1).flatten().cpu().numpy())
                fake_max_pred.extend(output_fake.max(-1).values.flatten().cpu().numpy())
                real_pred_g_mean_3.extend(output_real.pow(3).mean(-1).pow(1/3).flatten().cpu().numpy())
                fake_pred_g_mean_3.extend(output_fake.pow(3).mean(-1).pow(1/3).flatten().cpu().numpy())
                real_pred_g_mean_15.extend(output_real.pow(15).mean(-1).pow(1/15).flatten().cpu().numpy())
                fake_pred_g_mean_15.extend(output_fake.pow(15).mean(-1).pow(1/15).flatten().cpu().numpy())

                max_pred_reverse.extend((1 - output).max(-1).values.flatten().cpu().numpy())
                g_mean_pred_reverse.extend((1 - output).pow(3).mean(-1).pow(1/3).flatten().cpu().numpy())

                reals.extend(output_real.flatten())
                fakes.extend(output_fake.flatten())

        labels = torch.tensor(labels_)
        real_mean_pred = torch.tensor(real_mean_pred)
        real_max_pred = torch.tensor(real_max_pred)
        fake_mean_pred = torch.tensor(fake_mean_pred)
        fake_max_pred = torch.tensor(fake_max_pred)
        reals = torch.tensor(reals)
        fakes = torch.tensor(fakes)
    
        # Plot PR and AUC ROC
        os.makedirs(f"results/visualizations/{model_name}/{generator}", exist_ok=True)
        plot_auc_roc(labels_, output_max, plt.subplot(2, 3, 1), "Max")
        plot_auc_roc(labels_, output_mean, plt.subplot(2, 3, 2), "Mean")
        plot_auc_roc(labels_, output_g_mean_3, plt.subplot(2, 3, 3), "GeM, p=3")
        
        plot_pr(labels_, output_max, plt.subplot(2, 3, 4), "Max")
        plot_pr(labels_, output_mean, plt.subplot(2, 3, 5), "Mean")
        plot_pr(labels_, output_g_mean_3, plt.subplot(2, 3, 6), "GeM, p=3")
        plt.tight_layout()
        plt.savefig(os.path.join("results", "visualizations", model_name, generator, f"roc_pr_curves.png"))
        plt.close()

        # Plot PR and AUC ROC reverse
        plt.figure(figsize=(8, 8))
        plot_auc_roc(labels_, max_pred_reverse, plt.subplot(2, 3, 1), "Max(1-y)")
        plot_auc_roc(labels_, 1 - torch.tensor(output_mean), plt.subplot(2, 3, 2), "1-Mean")
        plot_auc_roc(labels_, g_mean_pred_reverse, plt.subplot(2, 3, 3), "GeM(1-y), p=3")
        
        plot_pr(labels_, max_pred_reverse, plt.subplot(2, 3, 4), "Max(1-y)")
        plot_pr(labels_, 1 - torch.tensor(output_mean), plt.subplot(2, 3, 5), "1-Mean")
        plot_pr(labels_, g_mean_pred_reverse, plt.subplot(2, 3, 6), "GeM(1-y), p=3")
        plt.tight_layout()
        plt.savefig(os.path.join("results", "visualizations", model_name, generator, f"roc_pr_curves_reverse.png"))
        plt.close()

        # Plot Prediction distribution
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.hist(real_mean_pred, bins=50, alpha=0.5, label="Real Mean")
        plt.hist(fake_mean_pred, bins=50, alpha=0.5, label="Fake Mean")
        plt.axvline(real_mean_pred.mean(), color='blue', linestyle='dashed', linewidth=1, label='Real Mean Mean')
        plt.axvline(fake_mean_pred.mean(), color='red', linestyle='dashed', linewidth=1, label='Fake Mean Mean')
        plt.title("Mean Prediction Distribution")
        plt.xlabel("Prediction Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.subplot(1, 4, 2)
        plt.hist(real_max_pred, bins=50, alpha=0.5, label="Real Max")
        plt.hist(fake_max_pred, bins=50, alpha=0.5, label="Fake Max")
        plt.axvline(real_max_pred.mean(), color='blue', linestyle='dashed', linewidth=1, label='Real Max Mean')
        plt.axvline(fake_max_pred.mean(), color='red', linestyle='dashed', linewidth=1, label='Fake Max Mean')
        plt.title("Maximum Prediction Distribution")
        plt.xlabel("Prediction Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.subplot(1, 4, 3)
        plt.hist(real_pred_g_mean_3, bins=50, alpha=0.5, label="Real Generalized mean")
        plt.hist(fake_pred_g_mean_3, bins=50, alpha=0.5, label="Fake Generalized mean")
        plt.axvline(np.mean(real_pred_g_mean_3), color='blue', linestyle='dashed', linewidth=1, label='Real Generalized Mean (p=3) average')
        plt.axvline(np.mean(fake_pred_g_mean_3), color='red', linestyle='dashed', linewidth=1, label='Fake Generalized Mean (p=3) average')
        plt.title("Generalized mean distribution, p=3")
        plt.xlabel("Prediction Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.subplot(1, 4, 4)
        plt.hist(real_pred_g_mean_15, bins=50, alpha=0.5, label="Real Generalized mean")
        plt.hist(fake_pred_g_mean_15, bins=50, alpha=0.5, label="Fake Generalized mean")
        plt.axvline(np.mean(real_pred_g_mean_15), color='blue', linestyle='dashed', linewidth=1, label='Real Generalized Mean (p=15) average')
        plt.axvline(np.mean(fake_pred_g_mean_15), color='red', linestyle='dashed', linewidth=1, label='Fake Generalized Mean (p=15) average')
        plt.title("Generalized mean distribution, p=15")
        plt.xlabel("Prediction Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("results", "visualizations", model_name, generator, f"prediction_distribution.png"))
        plt.close()

def make_contact_sheet(imagelist, ncolrow, textlist = None, textlist2 = None, labels = None, imsize = 100, mar = (5,5,5,5), padding = 5):
    """\
    Make a contact sheet from a list of filenames or images:

    imagelist    A list of filenames or images
    textlist     List of strings or numbers to be printing at the top part     
    textlist2     List of strings or numbers to be printing at the bottom part     
    labels       list of integers [0,16] defining the color of the border
    ncolrow      Number of columns and rows in the contact sheet
    imsize       Resize images to imsize x imsize
    mar          The left, top, right, bottom margin in pixels
    padding      The padding between images in pixels

    returns a PIL image object.
    """
    (marl,mart,marr,marb) = mar
    (ncols,nrows) = ncolrow

    # Read in all images and resize appropriately
    if isinstance(imagelist[0],str):
        imgs = [Image.open(fn).resize((imsize,imsize)) for fn in imagelist]
    else:
        imgs = imagelist
        
    if textlist is not None:
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 20)
        for i in range(len(imgs)):  
            # pdb.set_trace()
            # cur_im = imgs[i].convert('RGB')
            # pdb.set_trace()
            # d = ImageDraw.Draw(imgs[i].convert('RGB'))
            d = ImageDraw.Draw(imgs[i])
            try:
                d.text((5,5), str("%.4f" % textlist[i]), font=fnt, fill=(25, 255, 10))
            except:
                d.text((5,5), str("%.4f" % textlist[i]), font=fnt, fill=(255))

    if textlist2 is not None:
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 20)
        for i in range(len(imgs)):    
            d = ImageDraw.Draw(imgs[i])
            d.text((5,imsize-25), str(textlist2[i]), font=fnt, fill=(25, 255, 10))

    bordersize = 0
    if labels is not None:
        bordersize = 5
        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'brown', 'cyan', 'orange', 'purple', 'brown', 'lime', 'teal', 'navy', 'wheat', 'silver', 'dimgray', 'black']
        for i in range(len(imgs)):    
            imgs[i] = ImageOps.expand(imgs[i], border = bordersize, fill = colors[labels[i]])

    # Calculate the size of the output image, based on the
    #  photo thumb sizes, margins, and padding
    marw = marl+marr
    marh = mart+ marb

    padw = (ncols-1)*padding
    padh = (nrows-1)*padding
    imsize += 2*bordersize 
    isize = (ncols*imsize+marw+padw,nrows*imsize+marh+padh)

    # Create the new image. The background doesn't have to be white
    white = (255,255,255)
    inew = Image.new('RGB',isize,white)

    # Insert each thumb:
    for irow in range(nrows):
        for icol in range(ncols):
            left = marl + icol*(imsize+padding)
            right = left + imsize
            upper = mart + irow*(imsize+padding)
            lower = upper + imsize
            bbox = (left,upper,right,lower)
            try:
                img = imgs.pop(0)
            except:
                break
            inew.paste(img,bbox)
    return inew

def visualize_patchify(
        image, 
        patch_size=(224, 224), 
        stride=(16, 16),
        name="patchified_image",
    ):
    """
    Visualize the patchified image.
    """
    patches = patchify_image(image, patch_size, stride)

    h_stride, w_stride = stride
    h_crop, w_crop = patch_size
    batch_size, _, h_img, w_img = image.shape
    
    num_patches = patches.shape[1]
    print(f"Original image shape: {image.shape}")
    print(f"Patchified image shape: {patches.shape}")
    num_patches_h = floor(max(h_img - h_crop, 0) / h_stride + 1)
    num_patches_w = floor(max(w_img - w_crop, 0) / w_stride + 1)
    print(f"num_patches_h: {num_patches_h}, num_patches_w: {num_patches_w}")
    loss_h = image.shape[2] - (num_patches_h - 1) * stride[0] - patch_size[0]
    loss_w = image.shape[3] - (num_patches_w - 1) * stride[1] - patch_size[1]
    print(f"Pixels lost: {(loss_h * image.shape[3] + loss_w * image.shape[2] - loss_h * loss_w) / (image.shape[2] * image.shape[3]) * 100:.2f}%")
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
        # ax.set_title(f"Patch {i + 1}")
        ax.axis('off')
    plt.tight_layout()
    # Make margins for the figure 0 inches
    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
        wspace=0.1,  # control horizontal spacing between subplots
        hspace=0.1   # control vertical spacing between subplots
    )
    os.makedirs("results/visualizations", exist_ok=True)
    plt.savefig(os.path.join("results", "visualizations", name + ".png"))
    plt.close()

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    fig, axes = plt.subplots(
        nrows=h_grids,
        ncols=w_grids,
        figsize=(w_grids * 2, h_grids * 2),
    )
    print(f"Number of new patches: {h_grids * w_grids}")
    print(f"h_grids: {h_grids}, w_grids: {w_grids}")
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            im = transforms.ToPILImage()(image[:, :, y1:y2, x1:x2].squeeze(0))
            axes[h_idx, w_idx].imshow(im)
            axes[h_idx, w_idx].set_title(f"Patch {h_idx * w_grids + w_idx + 1}")
            axes[h_idx, w_idx].axis('off')
    plt.tight_layout()
    os.makedirs("results/visualizations", exist_ok=True)
    plt.savefig(os.path.join("results", "visualizations", "patchified_image_new.png"))
    plt.close()
    return patches

def visualize_patch_impact(
        img,
        model,
        stride,
        patch_size,
        ax,
        colorbar=False,
    ):
    """
    Visualizes the impact of each patch on the model's output.
    """
    model.eval()

    if img.dim() == 3:
        img = img.unsqueeze(0).to("cuda:0")
    
    _, _, h, w = img.shape
    n_h, n_w = h // patch_size, w // patch_size
    h_img, w_img = n_h * patch_size, n_w * patch_size
    img = img[:, :, :h_img, :w_img]
    with torch.no_grad():   
        output = model.forward_slide(img, reshape=False, stride=stride).sigmoid().squeeze(0)
        pred = output.mean()
        
    hotspot_image = np.zeros((h_img, w_img), dtype=np.float32)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            patch = output[i, j].expand(patch_size, patch_size)
            hotspot_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch.cpu().numpy()
    im = ax.imshow(hotspot_image, cmap="plasma", alpha=0.35, vmin=0, vmax=1)
    ax.text(0.5, 0.1, f"Prediction: {100 * pred.item():.2f}%",
                          ha='center', va='top', transform=ax.transAxes,
                            fontsize=10, bbox=dict(facecolor='white', alpha=0.35, edgecolor='none'))
    ax.axis("off")

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

def plot_image_prediction(
        json_path,
        model,
        transform,
        save_path,
        gen_name="default",
        patch_size=14,
        stride=112,
        ranges=(0, 0.1),
        num_images=15,
    ):
    plt.style.use('seaborn-v0_8')
    os.makedirs(f"results/visualizations/patch_impact/{save_path}", exist_ok=True)
    with open(json_path, "rb") as f:
        data = json.load(f)
    images = data["images"] 
    
    # Store selected images for each confidence level
    selected_images = []
    
    # Find one image for each confidence level
    for im in images:
        img_path = im["path"]
        label = im["label"]
        output = im["output"]

        if output > ranges[0] and output < ranges[1]:
            selected_images.append((img_path, label, output))

    selected_images.sort(key=lambda x: x[2])
    num_images = min(num_images, len(selected_images))
    # Plot for each selected image
    for j, image_data in enumerate(selected_images[:num_images]):
        print(f"Processing image {image_data[0]} {j + 1}/{num_images} for generator {gen_name}...")
        if image_data is None:
            continue
    
        img_path, label, output = image_data
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size
        max_dim = max(img_width, img_height)
        norm_width = img_width / max_dim
        norm_height = img_height / max_dim

        # Scale the figure: 4 columns (so width × 4), height stays
        scale = 4  # adjust for overall size (acts like a zoom factor)
        fig_width = scale * 2 * norm_width
        fig_height = 1.1 * scale * norm_height

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(1, 2, width_ratios=[1, 1.05])
                    
        # img_256 = transforms.Resize(256)(img)
        # img_512 = transforms.Resize(512)(img)
        
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(img)
        ax.set_title("Original image", fontsize=11, pad=5)
        ax.axis('off')
        ax.text(0.5, 0.1, f"Label: {label}",
                          ha='center', va='top', transform=ax.transAxes,
                          fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        # for i, (im, name) in enumerate(zip([img, img_512, img_256], ["Original", "Resize 512", "Resize 256"])):
        ax = fig.add_subplot(gs[0, 1])
        name = f"({img.size[0]}x{img.size[1]})"

        ax.imshow(img, alpha=0.5)
        ax.set_title(f"{name}", fontsize=10, pad=3)
        img = transform(img)

        visualize_patch_impact(
            img=img,
            model=model,
            stride=stride,
            patch_size=patch_size,
            ax=ax,
            colorbar=True
        )
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join("results", "visualizations", "patch_impact", save_path, f"image_prediction_{ranges[0]}_{j}_{gen_name}.png"))
        plt.close()
    print(f"Saved patch impact visualization to results/visualizations/patch_impact/{save_path}")

def save_predictions(
        experiment,
        model,
        gen_name,
        dl,
        device,
        **kwargs
    ):
    model.eval()
    print(f"Saving worst and best predictions for {gen_name}...")
    examples = {
        "images": [],
    }

    for data in tqdm.tqdm(dl, desc=f"Testing on generator {gen_name}", unit="batch"):
        images, labels, img_paths = data
        if isinstance(images, list):
            images = [im.float().to(device) for im in images]
        else:
            images = images.float().to(device)
            labels = labels.numpy().tolist()
        output = model.predict(images, **kwargs)
        output = np.atleast_1d(output).astype(np.float32)
        
        for i in range(len(labels)):
            examples["images"].append({
                "path": img_paths[i],
                "label": labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i],
                "output": output[i].item(),
            })

    examples["images"].sort(key=lambda x: x["output"], reverse=True)
    os.makedirs(f"results/predictions/{experiment['save_path']}", exist_ok=True)

    with open(f"results/predictions/{experiment['save_path']}/predictions_{gen_name}.json", "w") as f:
        json.dump(examples, f, indent=4)
    print(f"Saved worst and best predictions to results/predictions/{experiment['save_path']}")

if __name__ == "__main__":
    # img = Image.open("data/test/spai/stable-diffusion-3/1_fake/000005394_7.webp").convert("RGB")
    # patch_size = (512, 512)
    # stride = (256, 256)
    # patches = visualize_patchify(
    #     image=transforms.ToTensor()(img).unsqueeze(0),
    #     patch_size=patch_size,
    #     stride=stride,
    # )
    # print(patches.shape)
    # for i in range(patches.shape[1]):
    #     visualize_patchify(
    #         image=patches[0, i].unsqueeze(0),
    #         patch_size=(16, 16),
    #         stride=(16, 16),
    #         name=f"patch_{i}",
    #     )
    visualize_detection_curves(['WindowIntermediatePacth', 'Rine_all_classes', 'SPAI_progan'])

    # experiment = json.load(
    #     open(f"ckpt/IntermediatePatchLDM/2_nproj_1024_proj_dim/experiment.json", "rb")
    # )
    # model = IntermediatePatch(
    #     backbone=experiment["backbone"],
    #     nproj=experiment["nproj"],
    #     proj_dim=experiment["proj_dim"],
    #     device=torch.device("cuda:0"),
    # )
    # model.load_state_dict(
    #     torch.load(f"ckpt/IntermediatePatchLDM/2_nproj_1024_proj_dim/train.pth", map_location="cuda:0")
    # )

    # experiment = json.load(
    #     open(f"ckpt/RineModel/2_nproj_1024_proj_dim/experiment.json", "rb")
    # )
    # model = RineModel(
    #     backbone=experiment["backbone"],
    #     nproj=experiment["nproj"],
    #     proj_dim=experiment["proj_dim"],
    #     device=torch.device("cuda:0"),
    # )
    # model.load_state_dict(
    #     torch.load(f"ckpt/RineModel/2_nproj_1024_proj_dim/train.pth", map_location="cuda:0")
    # )
    # experiment = json.load(
    #     open(f"ckpt/IntermediatePatchSigLIP/3_nproj_512_proj_dim/experiment.json", "rb")
    # )
    # model = SigLIPIntermediate(
    #     backbone=experiment["backbone"],
    #     nproj=experiment["nproj"],
    #     proj_dim=experiment["proj_dim"],
    #     device=torch.device("cuda:0"),
    # )
    # model.load_state_dict(
    #     torch.load(f"ckpt/IntermediatePatchSigLIP/3_nproj_512_proj_dim/train.pth", map_location="cuda:0")
    # )

    # # tr = get_transform("val")
    # # test = get_loader(
    # #         experiment=experiment,
    # #         split="test",
    # #         transforms=tr,
    # #     )
    # # visualize_prediction_distribution(
    # #     data=test,
    # #     model=model,
    # #     device=torch.device("cuda:0"),
    # # )

    # device = "cuda:0"
    # transform = get_transform("no_crop_no_norm")
    # target = "fake"
    # for gen_name in get_generators():
    #     if not os.path.exists(f"results/predictions/{experiment['save_path']}/predictions_{gen_name.split('/')[-1]}.json"):
    #         print(f"Processing generator {gen_name.split('/')[-1]}...")
    #         loader = DataLoader(
    #                             EvaluationDataset(gen_name, transforms=transform, target=target),
    #                             batch_size=8,
    #                             shuffle=False,
    #                             pin_memory=True,
    #                             drop_last=False,
    #                             collate_fn=image_enlisting_collate_fn
    #                         )
    #         save_predictions(
    #             experiment=experiment,
    #             model=model,
    #             gen_name=gen_name.split("/")[-1],
    #             dl=loader,
    #             device=device,
    #             method="mean",
    #             window_slide=True
    #         )

    #         for rang in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1)]:
    #             plot_image_prediction(
    #                 json_path=f"results/predictions/{experiment['save_path']}/predictions_{gen_name.split('/')[-1]}.json",
    #                 model=model,
    #                 transform=transform,
    #                 gen_name=gen_name.split("/")[-1],
    #                 patch_size=14,
    #                 stride=112,
    #                 ranges=rang,
    #                 save_path=f"{experiment['save_path']}/fake",
    #                 num_images=3
    #             )
    #     else:
    #         print(f"Skipping {gen_name} as predictions already exist.")

    # target = "real"
    # for gen_name, real_name in get_generators(True): # get_generators()[:22]:
    #     if not os.path.exists(f"results/predictions/{experiment['save_path']}/predictions_{real_name}_real.json"):
    #         print(f"Processing generator {real_name}...")
    #         loader = DataLoader(
    #                             EvaluationDataset(gen_name, transforms=transform, target=target),
    #                             batch_size=8,
    #                             shuffle=False,
    #                             pin_memory=True,
    #                             drop_last=False,
    #                             collate_fn=image_enlisting_collate_fn
    #                         )
    #         print(f"Loaded {len(loader.dataset)} images from {real_name} for {target} target.")
    #         save_predictions(
    #             experiment=experiment,
    #             model=model,
    #             gen_name=f"{real_name}_real",
    #             dl=loader,
    #             device="cuda:0",
    #             method="mean",
    #             window_slide=True
    #         )

    #         for rang in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1)]:
    #             plot_image_prediction(
    #                 json_path=f"results/predictions/{experiment['save_path']}/predictions_{real_name}_real.json",
    #                 model=model,
    #                 transform=transform,
    #                 gen_name=real_name,
    #                 patch_size=14,
    #                 stride=112,
    #                 ranges=rang,
    #                 save_path=f"{experiment['save_path']}/real",
    #                 num_images=3
    #             )
    #     else:
    #         print(f"Skipping {real_name} as predictions already exist.")