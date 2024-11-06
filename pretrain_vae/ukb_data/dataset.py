"""This module contains the ECGdataset class, a custom Dataset for loading and
transforming ECG images stored in NPZ format.

Example
-------
    >>> from torchvision import transforms
    >>> transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((0.5,), (0.5,))
    >>> ])
    >>> dataset = ECGDataset(npz_dir='path/to/npz_files', transform=transform)
"""

import os
import numpy as np
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """Creates a Dataset of ECG images with self-defined
    transformations.

    Parameters
    ----------
    npz_dir : `str`
        Directory with npz files to be used.
    transform: callable
        A transform to apply on the images.

    """

    def __init__(self, npz_dir, transform):
        self.npz_dir = npz_dir
        self.transform = transform
        self.npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_file = os.path.join(self.npz_dir, self.npz_files[idx])

        # Load npz data from file
        data = np.load(npz_file)

        # Get image data
        array = data["image"]

        # Only take RGB channels
        image = array[:, :, :3]

        # Apply tranformations
        if self.transform:
            image = self.transform(image)
        return image
