"""
Script to visualize the reconstructions of a trained VAE on ECG data.

This script loads a pre-trained VAE model and uses it on a validation set of
ECG images, visualizing the reconstructions to assess the model's performance.
The visualizations are saved as side-by-side images comparing the original ECG
images with their reconstructions.

Examples
--------
To run the script and generate visualizations for a pre-trained model:
    $ python visualize_reconstructions.py

Notes
-----
The function requires 'DEVICE', 'VAL_DIR' and 'BEST_MODEL' to be defined in a
global 'config.py' file. The `visualize_reconstruction` function is used to
save visualizations to the disk in a directory named 'reconstructions' within
the script's directory, with filenames that include timestamps.
"""

import torch
from data.dataset import ECGDataset
from data.utils import get_transform
from torch.utils.data import DataLoader
from visualize import visualize_reconstruction
from config import DEVICE, VAL_DIR, BEST_MODEL

# Load the full model
model = torch.load(BEST_MODEL, map_location=torch.device("cpu"))
model.to(DEVICE)
print("Model created")
transform = get_transform()

# Create datasets
val_dataset = ECGDataset(VAL_DIR, transform=transform)
print("Datasets created")

# Create data loaders
val_loader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=False)
print("Dataloaders created")

# Visualize
visualize_reconstruction(model, val_loader)
