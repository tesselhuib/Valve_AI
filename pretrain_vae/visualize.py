"""Module to visualize reconstructions from a trained VAE model on ECG data.

This module provides a function, `visualize_reconstruction`, which generates
and saves a side-by-side plot of original and reconstructed ECG images.

Attributes
----------
script_dir : `str`
    The absolute path to the current script directory, used for saving
    reconstructions in the appropriate location on disk.

Example
-------
>>> from visualize import visualize_reconstruction
>>> visualize_reconstruction(trained_model, train_loader)
"""

import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from config import DEVICE

# For saving the reconstruction get the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))


def visualize_reconstruction(model, data_loader):
    """Visualizes the quality of reconstruction by the trained VAE model.

    It takes a batch of data from the DataLoader and performs a forward pass
    to generate reconstructions. Then, it plots a few reconstructions alongside
    their original images.

    Parameters
    ----------
    model: `torch.nn.Module`
        The trained VAE model.
    data_loader: `torch.utils.data.DataLoader`
        DataLoader containing the dataset from which a batch of data is sampled
        for reconstruction visualization.

    Notes
    -----
    The function saves the plot in the 'reconstructions' directory with
    a timestamped filename in PNG format. Ensure that this directory
    exists or is created before calling the function.
    """

    model.eval()
    with torch.no_grad():
        # Get a batch of data
        sample_data = next(iter(data_loader))[:5]
        sample_data = sample_data.to(DEVICE)

        # Generate reconstructions
        reconstructed, _, _ = model(sample_data)

        # Move tensors back to CPU for visualization
        sample_data = sample_data.cpu()
        reconstructed = reconstructed.cpu()

        # Plot original and reconstructed images
        n_images = 4
        fig, axes = plt.subplots(2, n_images, figsize=(15, 4))

        for i in range(n_images):
            # Original images
            axes[0, i].imshow(sample_data[i].squeeze(), cmap='gray')
            axes[0, i].set_title("Original")
            axes[0, i].axis('off')

            # Reconstructed images
            axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis('off')

        plt.savefig(os.path.join(script_dir, f'reconstructions/reconstruction_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.png'))
        plt.close(fig)
