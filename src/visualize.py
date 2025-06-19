import os
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from src.utils import patchify_image, get_transform, get_loader, image_enlisting_collate_fn
from src.data import EvaluationDataset
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pickle
from src.models import IntermediatePatch, SigLIPIntermediate
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, auc
import tqdm
import json
from torch.utils.data import DataLoader
from math import floor

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
    plt.style.use('seaborn-v0_8')
    fig, axs = plt.subplots(3, 2, figsize=(15, 20))
    suffixes = ['all', 'ldm', 'gan']

    for row_idx, suffix in enumerate(suffixes):
        arch_title = suffix.upper() + ' Models'
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

            # ROC Curve
            roc = curve_data['roc_curve']
            roc_auc = auc(roc[0], roc[1])
            axs[row_idx, 0].plot(
                roc[0], roc[1],
                lw=lw, alpha=alpha, zorder=zorder,
                label=f"{model} (AUC = {roc_auc:.2f})"
            )

            # Precision-Recall Curve
            precision, recall, _ = curve_data['precision_recall_curve']
            average_precision = auc(recall, precision)
            axs[row_idx, 1].plot(
                recall, precision,
                lw=lw, alpha=alpha, zorder=zorder,
                label=f"{model} (AP = {average_precision:.2f})"
            )

        # Format ROC subplot
        axs[row_idx, 0].set_title(f'ROC Curves - {arch_title}', fontsize=14)
        axs[row_idx, 0].set_xlabel('False Positive Rate', fontsize=12)
        axs[row_idx, 0].set_ylabel('True Positive Rate', fontsize=12)
        axs[row_idx, 0].plot([0, 1], [0, 1], 'k--', lw=1)
        axs[row_idx, 0].grid(True)
        axs[row_idx, 0].legend(loc='lower right', fontsize=10)

        # Format Precision-Recall subplot
        axs[row_idx, 1].set_title(f'Precision-Recall Curves - {arch_title}', fontsize=14)
        axs[row_idx, 1].set_xlabel('Recall', fontsize=12)
        axs[row_idx, 1].set_ylabel('Precision', fontsize=12)
        axs[row_idx, 1].set_ylim([0.0, 1.05])
        axs[row_idx, 1].grid(True)
        axs[row_idx, 1].legend(loc='lower left', fontsize=10)

    plt.tight_layout(pad=3.0)
    os.makedirs(f"results/visualizations/comparison", exist_ok=True)
    plt.savefig(os.path.join("results", "visualizations", 'comparison', f"comparison.png"))

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
        stride=(16, 16)
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
        ax.set_title(f"Patch {i + 1}")
        ax.axis('off')
    plt.tight_layout()
    os.makedirs("results/visualizations", exist_ok=True)
    plt.savefig(os.path.join("results", "visualizations", "patchified_image.png"))
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

def visualize_patch_impact(
        img,
        model,
        stride,
        patch_size,
        ax,
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
        
    hotspot_image = np.zeros((h_img, w_img), dtype=np.float32)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            patch = output[i, j].expand(patch_size, patch_size)
            hotspot_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch.cpu().numpy()
    ax.imshow(hotspot_image, cmap="plasma", alpha=0.5)
    ax.axis("off")

def plot_image_prediction(
        json_path,
        model,
        transform,
        device: torch.device,
        gen_name="default",
        patch_size=14,
        stride=112,
        ncols=6,
        nrows=4,
    ):
    plt.style.use('seaborn-v0_8')
    with open(json_path, "rb") as f:
        data = json.load(f)
    images = data["images"]

    checks = [False] * 6
    # Increased figure size and reduced spacing
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3.5))
    
    # Reduce spacing between subplots
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.92, wspace=0.15, hspace=0.25)    
    
    # Store selected images for each confidence level
    selected_images = [None] * 6
    
    # Find one image for each confidence level
    for im in images:
        img_path = im["path"]
        label = im["label"]
        output = im["output"]
        
        if all(checks):
            break

        if output < 0.1 and not checks[0]:
            checks[0] = True
            selected_images[0] = (img_path, label, output, "Very Low Confidence Prediction")
        elif output < 0.25 and output > 0.1 and not checks[1]:
            checks[1] = True
            selected_images[1] = (img_path, label, output, "Low Confidence Prediction")
        elif 0.25 <= output < 0.5 and not checks[2]:
            checks[2] = True
            selected_images[2] = (img_path, label, output, "Medium Confidence Prediction")
        elif 0.5 <= output < 0.75 and not checks[3]:
            checks[3] = True
            selected_images[3] = (img_path, label, output, "High Confidence Prediction")
        elif 0.75 <= output < 0.9 and not checks[4]:
            checks[4] = True
            selected_images[4] = (img_path, label, output, "Very High Confidence Prediction")
        elif output >= 0.9 and not checks[5]:
            checks[5] = True
            selected_images[5] = (img_path, label, output, "Extreme Confidence Prediction")

    # Plot for each selected image
    for col, image_data in enumerate(selected_images):
        print(f"Processing image {col + 1}/{ncols} for generator {gen_name}...")
        if image_data is None:
            continue
            
        img_path, label, output, title = image_data
        img = Image.open(img_path).convert("RGB")
        img_256 = transforms.Resize(256)(img)
        img_512 = transforms.Resize(512)(img)
        
        axes[0, col].imshow(img)
        axes[0, col].set_title(title, fontsize=11, pad=5)
        axes[0, col].axis('off')
        axes[0, col].text(0.5, -0.1, f"Label: {label}, Output: {output:.4f}",
                          ha='center', va='top', transform=axes[0, col].transAxes,
                          fontsize=9, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        for i, (im, name) in enumerate(zip([img, img_512, img_256], ["Original", "512x512", "256x256"])):
            axes[i + 1, col].imshow(im)
            axes[i + 1, col].set_title(f"{name}", fontsize=9, pad=3)
            im = transform(im)

            visualize_patch_impact(
                img=im,
                model=model,
                stride=stride,
                patch_size=patch_size,
                ax=axes[i + 1, col],
            )

    # Hide unused subplots
    for col in range(ncols):
        if selected_images[col] is None:
            for row in range(nrows):
                axes[row, col].axis('off')

    os.makedirs("results/visualizations/patch_impact", exist_ok=True)
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join("results", "visualizations", "patch_impact", f"image_prediction_{gen_name}.png"))
    print(f"Saved patch impact visualization to results/visualizations/patch_impact/image_prediction_{gen_name}.png")

def save_predictions(
        experiment,
        model,
        gen_name,
        dl,
        device,
        **kwargs
    ):
    model.eval()
    gen_name = gen_name.split("/")[-1]
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
    # visualize_detection_curves(['IntermediatePatch', 'Rine_4_class', 'Rine_latent_diffusion', 'SPAI'])
    
    gen_name = "synthbuster/midjourney-v5"

    experiment = json.load(
        open(f"ckpt/IntermediatePatch/2_nproj_1024_proj_dim/experiment.json", "rb")
    )
    model = IntermediatePatch(
        backbone=experiment["backbone"],
        nproj=experiment["nproj"],
        proj_dim=experiment["proj_dim"],
        device=torch.device("cuda:0"),
    )
    model.load_state_dict(
        torch.load(f"ckpt/IntermediatePatch/2_nproj_1024_proj_dim/train.pth", map_location="cuda:0")
    )

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
    # stride = 128
    # patch_size = 16
    # tr = transforms.Resize(256)
    # tr = lambda x: x

    # visualize_image_impact(
    #     img=tr(img).unsqueeze(0).to("cuda:0"),
    #     model=model,
    # )

    # generator = "progan"
    # dl = torch.utils.data.DataLoader(
    #     dataset=EvaluationDataset(
    #         generator=generator,
    #         transforms=tr,
    #     ),
    #     batch_size=64,
    #     shuffle=False,
    #     num_workers=2,
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

    device = "cuda:0"
    transform = get_transform("no_crop_no_norm")

    plot_image_prediction(
        json_path=f"results/predictions/{experiment['save_path']}/predictions_{gen_name.split('/')[-1]}.json",
        model=model,
        transform=transform,
        device=device,
        gen_name=gen_name.split("/")[-1],
        patch_size=14,
        stride=112,
    )

    # loader = DataLoader(
    #                     EvaluationDataset(gen_name, transforms=transform, target="fake"),
    #                     batch_size=8,
    #                     shuffle=False,
    #                     pin_memory=True,
    #                     drop_last=False,
    #                     collate_fn=image_enlisting_collate_fn
    #                 )

    # save_predictions(
    #     experiment=experiment,
    #     model=model,
    #     gen_name=gen_name,
    #     dl=loader,
    #     device="cuda:0",
    #     method="mean",
    #     window_slide=True
    # )