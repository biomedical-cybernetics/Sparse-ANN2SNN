import random
import numpy as np
import torch
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def get_imagenet(path, nsamples=128, batch_size=128, seed=0):
    """
    Load a small subset of ImageNet for calibration.
    """
    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    # Assuming 'val' folder exists at path. Use 'train' if calibrating on train set.
    try:
        dataset = datasets.ImageFolder(root=path, transform=transform)
    except:
        print(f"Error loading ImageNet from {path}. Please check path.")
        return []

    # Randomly select nsamples indices
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), nsamples, replace=False)

    # Create a subset sampler (or just manually extract)
    # Manual extraction to ensure exact count and easy batching
    images = []
    for i in indices:
        img, _ = dataset[i]
        images.append(img.unsqueeze(0)) # Add batch dim

    # Concatenate into larger batches if batch_size > 1
    # For RIA logic, usually we process in the loop with nsamples,
    # but prune_ria implementation often feeds inputs one by one or in blocks.
    # Here we return a list of tensors on device

    calibration_data = []
    current_batch = []

    for img in images:
        current_batch.append(img)
        if len(current_batch) == batch_size:
            calibration_data.append(torch.cat(current_batch, dim=0))
            current_batch = []

    if len(current_batch) > 0:
        calibration_data.append(torch.cat(current_batch, dim=0))

    return calibration_data
