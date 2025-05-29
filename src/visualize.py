import os
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from src.utils import patchify_image, get_transform, get_loader
from src.data import EvaluationDataset
from PIL import Image
import pickle
from src.models import IntermediatePatch
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, precision_recall_curve, auc, average_precision_score
import tqdm
from torch.utils.data import DataLoader
from math import floor

def visualize_prediction_distribution(
        data,
        model,
        device: torch.device,
        model_name="IntermediatePatch",
    ):
    """
    Visualizes the prediction distribution of the model.
    """
    for generator, dl in data:
        generator = generator.split("/")[-1]
        real_mean_pred = []
        real_max_pred = []
        fake_mean_pred = []
        fake_max_pred = []
        real_pred_product = []
        fake_pred_product = []
        max_pred_reverse = []
        reals = []
        fakes = []
        labels_ = []
        output_mean =[]
        output_max = []
        real_diffs = []
        fake_diffs = []

        print(f'Fake: {len(dl.dataset.fake)}, Real: {len(dl.dataset.real)}, Total: {len(dl.dataset.images)}')
        with torch.no_grad():
            for data in tqdm.tqdm(dl, desc=f"Extracting tokens for {generator}", unit="batch"):
                images, labels, _ = data
                images, labels = images.float().to(device), labels.to(device)
                output, _ = model(images)
                output = output.sigmoid()

                output_mean.extend(output.mean(-1).flatten().cpu().numpy())
                output_max.extend(output.max(-1).values.flatten().cpu().numpy())
                labels_.extend(labels.cpu().numpy())

                output_fake = output[labels == 1]
                output_real = output[labels == 0]
                real_mean_pred.extend(output_real.mean(-1).flatten().cpu().numpy())
                real_max_pred.extend(output_real.max(-1).values.flatten().cpu().numpy())
                fake_mean_pred.extend(output_fake.mean(-1).flatten().cpu().numpy())
                fake_max_pred.extend(output_fake.max(-1).values.flatten().cpu().numpy())
                real_pred_product.extend(output_real.prod(-1).flatten().cpu().numpy())
                fake_pred_product.extend(output_fake.prod(-1).flatten().cpu().numpy())

                max_pred_reverse.extend((1 - output).max(-1).values.flatten().cpu().numpy())
                real_diffs.extend((output_real.max(-1).values - output_real.min(-1).values).flatten().cpu().numpy())
                fake_diffs.extend((output_fake.max(-1).values - output_fake.min(-1).values).flatten().cpu().numpy())
                reals.extend(output_real.flatten())
                fakes.extend(output_fake.flatten())

        roc_curve_max = roc_curve(labels_, output_max)
        auc_max = auc(roc_curve_max[0], roc_curve_max[1])

        roc_curve_max_reverse = roc_curve(labels_, max_pred_reverse)
        auc_max_reverse = auc(roc_curve_max_reverse[0], roc_curve_max_reverse[1])

        roc_curve_mean = roc_curve(labels_, output_mean)
        auc_mean = auc(roc_curve_mean[0], roc_curve_mean[1])

        roc_curve_mean_reverse = roc_curve(labels_, 1 - torch.tensor(output_mean))
        auc_mean_reverse = auc(roc_curve_mean_reverse[0], roc_curve_mean_reverse[1])

        precision_recall_curve_max = precision_recall_curve(labels_, output_max)
        ap_max = average_precision_score(labels_, output_max)
        precision_recall_curve_max_reverse = precision_recall_curve(labels_, max_pred_reverse)
        ap_max_reverse = average_precision_score(labels_, max_pred_reverse)

        precision_recall_curve_mean = precision_recall_curve(labels_, output_mean)
        ap_mean = average_precision_score(labels_, output_mean)
        precision_recall_curve_mean_reverse = precision_recall_curve(labels_, 1 - torch.tensor(output_mean))
        ap_mean_reverse = average_precision_score(labels_, 1 - torch.tensor(output_mean))

        labels = torch.tensor(labels_)
        real_mean_pred = torch.tensor(real_mean_pred)
        real_max_pred = torch.tensor(real_max_pred)
        fake_mean_pred = torch.tensor(fake_mean_pred)
        fake_max_pred = torch.tensor(fake_max_pred)
        real_diffs = torch.tensor(real_diffs)
        fake_diffs = torch.tensor(fake_diffs)
        reals = torch.tensor(reals)
        fakes = torch.tensor(fakes)
    
        os.makedirs(f"results/visualizations/{model_name}/{generator}", exist_ok=True)
        plt.figure(figsize=(14, 6))
        roc_curve_display = RocCurveDisplay(
            fpr=roc_curve_max[0],
            tpr=roc_curve_max[1],
            roc_auc=auc_max,
            estimator_name="Max",
        )
        roc_curve_display.plot(ax=plt.subplot(2, 4, 1))
        plt.subplot(2, 4, 1).set_title(f"ROC Curve Max (AUC = {auc_max:.4f})")
        roc_curve_display = RocCurveDisplay(
            fpr=roc_curve_mean[0],
            tpr=roc_curve_mean[1],
            roc_auc=auc_mean,
        )
        roc_curve_display.plot(ax=plt.subplot(2, 4, 2))
        plt.subplot(2, 4, 2).set_title(f"ROC Curve Mean (AUC = {auc_mean:.4f})")
        precision_recall_display = PrecisionRecallDisplay(
            precision=precision_recall_curve_max[0],
            recall=precision_recall_curve_max[1],
            average_precision=ap_max,
        )
        precision_recall_display.plot(ax=plt.subplot(2, 4, 3))
        plt.subplot(2, 4, 3).set_title(f"PR Curve Max (AP = {ap_max:.4f})")
        precision_recall_display = PrecisionRecallDisplay(
            precision=precision_recall_curve_mean[0],
            recall=precision_recall_curve_mean[1],
            average_precision=ap_mean,
        )
        precision_recall_display.plot(ax=plt.subplot(2, 4, 4))
        plt.subplot(2, 4, 4).set_title(f"PR Curve Mean (AP = {ap_mean:.4f})")

        roc_curve_display = RocCurveDisplay(
            fpr=roc_curve_max_reverse[0],
            tpr=roc_curve_max_reverse[1],
            roc_auc=auc_max_reverse,
        )
        roc_curve_display.plot(ax=plt.subplot(2, 4, 5))
        plt.subplot(2, 4, 5).set_title(f"ROC Curve Max(1-y) (AUC = {auc_max_reverse:.4f})")
        roc_curve_display = RocCurveDisplay(
            fpr=roc_curve_mean_reverse[0],
            tpr=roc_curve_mean_reverse[1],
            roc_auc=auc_mean_reverse,
        )
        roc_curve_display.plot(ax=plt.subplot(2, 4, 6))
        plt.subplot(2, 4, 6).set_title(f"ROC Curve 1-Mean (AUC = {auc_mean_reverse:.4f})")
        precision_recall_display = PrecisionRecallDisplay(
            precision=precision_recall_curve_max_reverse[0],
            recall=precision_recall_curve_max_reverse[1],
            average_precision=ap_max_reverse,
        )
        precision_recall_display.plot(ax=plt.subplot(2, 4, 7))
        plt.subplot(2, 4, 7).set_title(f"PR Curve Max(1-y) (AP = {ap_max_reverse:.4f})")
        precision_recall_display = PrecisionRecallDisplay(
            precision=precision_recall_curve_mean_reverse[0],
            recall=precision_recall_curve_mean_reverse[1],
            average_precision=ap_mean_reverse,
        )
        precision_recall_display.plot(ax=plt.subplot(2, 4, 8))
        plt.subplot(2, 4, 8).set_title(f"PR Curve 1-Mean (AP = {ap_mean_reverse:.4f})")
        plt.tight_layout()
        plt.savefig(os.path.join("results", "visualizations", model_name, generator, f"roc_pr_curves.png"))
        plt.close()

        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.hist(real_mean_pred, bins=50, alpha=0.5, label="Real Mean")
        plt.hist(fake_mean_pred, bins=50, alpha=0.5, label="Fake Mean")
        plt.axvline(real_mean_pred.mean(), color='blue', linestyle='dashed', linewidth=1, label='Real Mean Mean')
        plt.axvline(fake_mean_pred.mean(), color='red', linestyle='dashed', linewidth=1, label='Fake Mean Mean')
        plt.title("Prediction Distribution")
        plt.xlabel("Prediction Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.subplot(1, 4, 2)
        plt.hist(real_max_pred, bins=50, alpha=0.5, label="Real Max")
        plt.hist(fake_max_pred, bins=50, alpha=0.5, label="Fake Max")
        plt.axvline(real_max_pred.mean(), color='blue', linestyle='dashed', linewidth=1, label='Real Max Mean')
        plt.axvline(fake_max_pred.mean(), color='red', linestyle='dashed', linewidth=1, label='Fake Max Mean')
        plt.title("Prediction Distribution")
        plt.xlabel("Prediction Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.subplot(1, 4, 3)
        plt.hist(real_pred_product, bins=50, alpha=0.5, label="Real Product")
        plt.hist(fake_pred_product, bins=50, alpha=0.5, label="Fake Product")
        plt.axvline(np.mean(real_pred_product), color='blue', linestyle='dashed', linewidth=1, label='Real Product Mean')
        plt.axvline(np.mean(fake_pred_product), color='red', linestyle='dashed', linewidth=1, label='Fake Product Mean')
        plt.title("Prediction Product Distribution")
        plt.xlabel("Prediction Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.subplot(1, 4, 4)
        plt.hist(real_diffs, bins=50, alpha=0.5, label="Real Diff")
        plt.hist(fake_diffs, bins=50, alpha=0.5, label="Fake Diff")
        plt.axvline(real_diffs.mean(), color='blue', linestyle='dashed', linewidth=1, label='Real Mean Diff Mean')
        plt.axvline(fake_diffs.mean(), color='red', linestyle='dashed', linewidth=1, label='Fake Mean Diff Mean')
        plt.title("Prediction difference (Max - Min)")
        plt.xlabel("Prediction Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("results", "visualizations", model_name, generator, f"prediction_distribution.png"))
        plt.close()

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
        image: Image.Image,
        transform,
        model,
    ):
    """
    Visualizes the impact of each patch on the model's output.
    """
    crop = transforms.RandomCrop(224)(image)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])(crop)
    # img = transform(image)

    if img.dim() == 3:
        img = img.unsqueeze(0).to("cuda:0")
    
    with torch.no_grad():   
        output = model(img)[0]

    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input Image")
    plt.subplot(1, 2, 2)
    # plt.imshow(img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
    plt.imshow(crop)
    plt.axis("off")
    plt.title("Transformed Image")
    output = output.sigmoid().squeeze(0)
    hotspot_image = np.zeros((224, 224), dtype=np.float32)
    for i in range(16):
        for j in range(16):
            k = i * 16 + j
            patch = output[k].expand(14, 14)
            hotspot_image[i * 14:(i + 1) * 14, j * 14:(j + 1) * 14] = patch.cpu().numpy()
    plt.imshow(hotspot_image, cmap="plasma", alpha=0.5)
    plt.colorbar(label="Patch Impact")
    plt.axis("off")
    plt.title("Model Output")
    plt.tight_layout()
    plt.savefig("results/visualizations/patch_impact_sd3.png")

def visualize_image_impact(
        img: torch.Tensor,
        model,
        **kwargs
    ) -> None:
    """
    Visualizes the impact of each pixel on the model's output.
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
    plt.imshow(gradients[0].detach().cpu().numpy(), cmap="viridis", alpha=0.5)
    plt.axis("off")
    plt.title("Gradient Impact")
    plt.savefig(f"gradients.png")

def save_worst_predictions(
        experiment,
        model,
        gen_name,
        dl,
        threshold,
        device,
        max_n=10,
        **kwargs
    ):
    model.eval()
    false_positives = []
    false_negatives = []
    for data in tqdm.tqdm(dl, desc=f"Testing on generator {gen_name}", unit="batch"):
        images, labels, img_paths = data
        images, labels = images.float().to(device), labels.to(device)
        output = model.predict(images, **kwargs)
        
        for i in range(len(labels)):
            if labels[i] == 0 and output[i] > threshold:
                false_positives.append((img_paths[i], output[i]))
            elif labels[i] == 1 and output[i] < threshold:
                false_negatives.append((img_paths[i], output[i]))
    print(f"False positives: {len(false_positives)}, False negatives: {len(false_negatives)}")

    os.makedirs(f"results/train/{experiment['save_path']}/worst_predictions/{gen_name}/", exist_ok=True)
    with open(f"results/train/{experiment['save_path']}/worst_predictions/{gen_name}/false_positives.txt", "w") as f:
        for i in range(min(max_n, len(false_positives))):
            img_path, score = false_positives[i]
            f.write(f"{img_path} {score:2.1f}\n")
    
    with open(f"results/train/{experiment['save_path']}/worst_predictions/{gen_name}/false_negatives.txt", "w") as f:
        for i in range(min(max_n, len(false_negatives))):
            img_path, score = false_negatives[i]
            f.write(f"{img_path} {score:2.1f}\n")
    print(f"Saved worst predictions to results/train/{experiment['save_path']}/worst_predictions/{gen_name}/")

def save_best_predictions(
        experiment,
        model,
        gen_name,
        dl,
        threshold,
        device,
        max_n=10,
        **kwargs
    ):
    model.eval()
    true_positives = []
    true_negatives = []
    for data in tqdm.tqdm(dl, desc=f"Testing on generator {gen_name}", unit="batch"):
        images, labels, img_paths = data
        images, labels = images.float().to(device), labels.to(device)
        output = model.predict(images, **kwargs)
        
        for i in range(len(labels)):
            if labels[i] == 0 and output[i] < threshold:
                true_positives.append((img_paths[i], output[i]))
            elif labels[i] == 1 and output[i] > threshold:
                true_negatives.append((img_paths[i], output[i]))
    print(f"True positives: {len(true_positives)}, True negatives: {len(true_negatives)}")

    os.makedirs(f"results/train/{experiment['save_path']}/best_predictions/{gen_name}/", exist_ok=True)
    with open(f"results/train/{experiment['save_path']}/best_predictions/{gen_name}/true_positives.txt", "w") as f:
        for i in range(min(max_n, len(true_positives))):
            img_path, score = true_positives[i]
            f.write(f"{img_path} {score:2.1f}\n")
    
    with open(f"results/train/{experiment['save_path']}/best_predictions/{gen_name}/true_negatives.txt", "w") as f:
        for i in range(min(max_n, len(true_negatives))):
            img_path, score = true_negatives[i]
            f.write(f"{img_path} {score:2.1f}\n")
    print(f"Saved best predictions to results/train/{experiment['save_path']}/best_predictions/{gen_name}/")


if __name__ == "__main__":
    # Load a sample image
    # image_path = os.path.join("data", "test", "progan", "airplane", "1_fake", "00069.png")
    # image = Image.open(image_path).convert("RGB")

    # image = transforms.ToTensor()(image)  # Convert to tensor
    # image = image.unsqueeze(0)  # Add batch dimension
    # visualize_patchify(image, patch_size=(224, 224), stride=(112, 112))
    # print("Patchified image saved to results/visualizations/patchified_image.png")

    # img = Image.open("data/test/diffusion_datasets/dalle/1_fake/efgchmasis.png")
    img = Image.open("data/test/spai/stable-diffusion-3/1_fake/000006315_7.webp")
    # img = Image.open("data/test/diffusion_datasets/laion/0_real/uyfkdpzowl.jpg")
    img = img.convert("RGB")
    tr = get_transform("random_crop")

    experiment = pickle.load(
        open(f"ckpt/IntermediatePatch/3_nproj_512_proj_dim/experiment.pickle", "rb")
    )
    model = IntermediatePatch(
        backbone=experiment["backbone"],
        nproj=experiment["nproj"],
        proj_dim=experiment["proj_dim"],
        device=torch.device("cuda:0"),
    )
    model.load_state_dict(
        torch.load(f"ckpt/IntermediatePatch/3_nproj_512_proj_dim/train.pth", map_location="cuda:0")
    )

    visualize_patch_impact(
        image=img,
        transform=tr,
        model=model,
    )

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

    # test = get_loader(
    #         experiment=experiment,
    #         split="test",
    #         transforms=tr,
    #     )
    # visualize_prediction_distribution(
    #     data=test,
    #     model=model,
    #     device=torch.device("cuda:0"),
    # )

    # device = "cuda:0"
    # transform = get_transform("val")
    # g = "stable-diffusion-3"
    # loader = DataLoader(
    #                     EvaluationDataset(g, transforms=transform, target="both"),
    #                     batch_size=16,
    #                     shuffle=False,
    #                     pin_memory=True,
    #                     drop_last=False,
    #                 )

    # save_worst_predictions(
    #     experiment=experiment,
    #     model=model,
    #     gen_name=g,
    #     dl=loader,
    #     device=device,
    #     threshold=0.5,
    #     method="max",
    # )

    # save_best_predictions(
    #     experiment=experiment,
    #     model=model,
    #     gen_name=g,
    #     dl=loader,
    #     device=device,
    #     threshold=0.5,
    #     method="max",
    # )