"""Script to generate latent embeddings from ECG data. A pretrained VAE is used
on the AUMC data to generate the latent embeddings.

Attributes
----------
PRETRAINED_MODEL : `str`
    The path to the pretrained model
TRAIN_DIR : `str`
    The path to the training set directory
VAL_DIR : `str`
    The path to the validation set directory
TEST_DIR : `str`
    The path to the test set directory
TRAIN_LABELS: `str`
    The path to the csv file with the labels for the training set
VAL_LABELS: `str`
    The path to the csv file with the labels for the validation set
TEST_LABELS: `str`
    The path to the csv file with the labels for the test set
BATCH_SIZE : `int`
    The batch size to process at once
DEVICE : `str`
    The device to use. Either 'cuda' or 'cpu'.

Notes
-----
The embeddings are saved to 'embedding_classifier/embeddings'. All the
attributes are expected to be loaded in from a global 'config.py' file.

"""

import os
import torch
import numpy as np
import pandas as pd
from data.ECGDataset import ECGDataset
from data.utils import get_transform
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import (
    PRETRAINED_MODEL,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    TRAIN_LABELS,
    VAL_LABELS,
    TEST_LABELS,
    BATCH_SIZE,
    DEVICE
)


def load_pretrained_encoder(device):
    """Loads the pretrained encoder

    Parameters
    ----------
    device : `str`
        The device to use. Either 'cuda' or 'cpu'.

    Returns
    -------
    torch.nn.Module
        The pretrained encoder set to evaluation mode.
    """

    vae = torch.load(PRETRAINED_MODEL, map_location=device)
    vae.eval()
    encoder = vae.encoder
    return encoder


def save_embeddings_to_csv(embeddings, labels, file_paths, filename):
    """Save generated embeddings to a csv file.

    Parameters
    ----------
    embeddings : numpy.Array
        A numpy array with the latent embeddings for all files
    labels : numpy.Array
        A numpy array with the labels for each file
    file_paths : `list [`str`]
        A list of strings of the filepaths for each npz file
    filename : `str`
        The filename of the csv file being created.
    """

    df = pd.DataFrame(
        embeddings, columns=[f"latent_{i}" for i in range(embeddings.shape[1])]
    )
    df.insert(0, "file_path", file_paths)
    df.insert(1, "label", labels)
    df.to_csv(filename, index=False)


def generate_and_save_embeddings(encoder, dataloader, filename, device):
    """Generates and saves embeddings in batches

    Parameters
    ----------
    encoder : torch.nn.Module
        The pretrained encoder set to evaluation mode.
    dataloader : torch.utils.data.DataLoader
        A DataLoader with the data to process
    filename : `str`
        The filename to save the embeddings as.
    device : `str`
        The device to use. Either 'cuda' or 'cpu'.
    """

    encoder.eval()
    all_embeddings = []
    all_labels = []
    all_file_paths = []

    for batch_data, batch_labels, batch_paths in tqdm(
        dataloader, desc=f"Generating embeddings for {filename}", mininterval=300
    ):
        with torch.no_grad():
            # Move data to device and pass batch through encoder
            batch_data = batch_data.to(device)
            mean, log_var = encoder(batch_data)

        # Move embeddings back to CPU and collect labels/paths
        all_embeddings.append(mean.cpu().numpy())
        all_labels.append(batch_labels.numpy())
        all_file_paths.extend(batch_paths)

    # Convert lists of arrays into single numpy arrays
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Save embeddings, labels, and file paths to CSV
    save_embeddings_to_csv(embeddings, labels, all_file_paths, filename)
    print(f"Saved {filename}, shape: {embeddings.shape}")


def load_dataset_and_dataloader(data_dir, labels_file, transform, batch_size):
    """Loads the dataset and dataloader for each split.

    Parameters
    ----------
    data_dir : `str`
        The path to the data directory.
    labels_file : `str`
        The path to the label file.
    transform : callable
        A transform to apply on the images.
    batch_size : `int`
        The batch size to process at once.

    Returns
    -------
    torch.utils.data.DataLoader
        A DataLoader instance for the data.
    """

    dataset = ECGDataset(data_dir, labels_file, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def main():
    # Device handling (GPU if available, otherwise CPU)
    device = DEVICE

    # Load the pretrained VAE encoder
    encoder = load_pretrained_encoder(device)

    # Load the dataset and dataloaders
    transform = get_transform()

    # For saving the reconstruction get the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Training dataset
    train_dataloader = load_dataset_and_dataloader(
        TRAIN_DIR, TRAIN_LABELS, transform, BATCH_SIZE
    )
    generate_and_save_embeddings(
        encoder,
        train_dataloader,
        os.path.join(script_dir, "embeddings/train_embeddings.csv"),
        device,
    )

    # Validation dataset
    val_dataloader = load_dataset_and_dataloader(
        VAL_DIR, VAL_LABELS, transform, BATCH_SIZE
    )
    generate_and_save_embeddings(
        encoder,
        val_dataloader,
        os.path.join(script_dir, "embeddings/val_embeddings.csv"),
        device,
    )

    # Test dataset
    test_dataloader = load_dataset_and_dataloader(
        TEST_DIR, TEST_LABELS, transform, BATCH_SIZE
    )
    generate_and_save_embeddings(
        encoder,
        test_dataloader,
        os.path.join(script_dir, "embeddings/test_embeddings.csv"),
        device,
    )


if __name__ == "__main__":
    main()
