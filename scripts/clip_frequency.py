import clip
import torch
import os
from src.data import TrainingDataset, EvaluationDataset
from src.utils import get_transform, get_loader, save_worst_predictions, find_best_acc_threshold

model, preprocess = clip.load("ViT-L/14", device="cuda:0" if torch.cuda.is_available() else "cpu")


