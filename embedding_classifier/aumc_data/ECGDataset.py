"""This module contains the ECGdataset class, a custom Dataset for loading and
transforming ECG images stored in NPZ format. It deals with the NPZ directory
structure for the AUMC data: patient ids are directories with their ECGs grouped
within these directories.

Example
-------
    >>> from torchvision import transforms
    >>> transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((0.5,), (0.5,))
    >>> ])
    >>> dataset = ECGDataset(npz_dir='path/to/npz_files', transform=transform)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd


class ECGDataset(Dataset):
    """Creates a Dataset of ECG images with self-defined
    transformations.

    Parameters
    ----------
    data_dir : `str`
        Directory with patient id subdirecties in which NPZ files are saved.
    label_csv: `str`
        Path to a csv file that contains the label for each patient id.
    transform: callable
        A transform to apply on the images.

    """

    def __init__(self, data_dir, label_csv, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.npz_files = []
        self.labels = []

        # Read label file and create dictionary mapping patient id to label
        df_labels = pd.read_csv(label_csv)
        label_dict = {row['Pseudo_id']: row['Label'] for _, row in df_labels.iterrows()}

        # Traverse through patient directories and get file paths and labels
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".npz"):
                    patient_id = int(os.path.basename(root))
                    if patient_id in label_dict:
                        file_path = os.path.join(root, file)
                        self.npz_files.append(file_path)
                        self.labels.append(label_dict[patient_id])

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        try:
            data = np.load(npz_file)
            ecg_data = data['image']
            label = self.labels[idx]

            # Only take RGB, not A
            ecg_data = ecg_data[:, :, :3]

            if self.transform:
                ecg_data = self.transform(ecg_data)

            label = torch.tensor(label, dtype=torch.long)
            return ecg_data, label, npz_file
        except EOFError:
            print(f"EOFError: No data left in file {npz_file}. Skipping this file.")
            return None
