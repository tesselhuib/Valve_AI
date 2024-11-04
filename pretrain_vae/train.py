"""
Module to train a Variational Autoencoder (VAE) on ECG data saved in NPZ format.

This script includes data preparation, model training with early stopping,
validation, and optional visualization of input vs. reconstructed images.

Functions
---------
prepare_dataloaders()
    Loads NPZ files into training, validation, and test DataLoaders.

loss_function(x, x_hat, mean, log_var)
    Computes the VAE loss, combining Binary Cross-Entropy and KL Divergence.

train(model, optimizer, train_loader, val_loader, patience=10)
    Trains the VAE model with an early stopping strategy.

validate(model, val_loader)
    Performs validation to compute the average loss on the validation set.

Notes
-----
The configuration values (such as directories, batch size and learning rate)
are loaded from a global configuration file `config.py`.
"""
import os
import time
import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader
from datetime import datetime

from models.vae import VAE
from data.dataset import ECGDataset
from data.utils import get_transform
from visualize import visualize_reconstruction
from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    LATENT_DIM,
    DEVICE,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    VISUALIZE_RECONSTRUCTION,
)

# For saving the best_model get the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))


def prepare_dataloaders():
    """Loads the NPZs into training, validation and test DataLoaders.


    Returns
    -------
    train_loader : `torch.utils.data.DataLoader`
        Dataloader for the training set.
    val_loader : `torch.utils.data.DataLoader`
        Dataloader for the validation set.
    test_loader : `torch.utils.data.DataLoader`
        Dataloader for the test set.

    Notes
    ----
    The function assumes a global configuration file, 'config.py', which
    defines the data directories 'TRAIN_DIR', 'VAL_DIR' and 'TEST_DIR'
    as well as the batch size 'BATCH_SIZE'.

    Examples
    --------
    >>> train_loader, val_loader, test_loader = prepare_dataloaders()
    """

    transform = get_transform()
    print("Creating datasets...", flush=True)
    train_dataset = ECGDataset(TRAIN_DIR, transform=transform)
    val_dataset = ECGDataset(VAL_DIR, transform=transform)
    test_dataset = ECGDataset(TEST_DIR, transform=transform)

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def loss_function(x, x_hat, mean, log_var):
    """Defines the loss function for a variational autoencoder, which includes
    Binary Cross-Entropy and KL Divergence loss.

    Parameters
    ----------
    x : `torch.Tensor`
        Input image tensor.
    x_hat : `torch.Tensor`
        Reconstructed output image tensor.
    mean : `torch.Tensor`
        Mean of the latent space distribution.
    log_var : `torch.Tensor`
        Logarithm of the variance of the latent space distribution.

    Returns
    -------
    total_loss : `torch.Tensor`
        The total loss used for training the VAE.
    """
    bce_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kld_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    total_loss = bce_loss + kld_loss
    return total_loss


def train(model, optimizer, train_loader, val_loader, patience=10):
    """Train the model.

    Parameters
    ----------
    model : `torch.nn.Module`
        The VAE model to be trained.
    optimizer : `torch.optim.Optimizer`
        The optimizer used for updating model weights.
    train_loader : `torch.utils.data.DataLoader`
        DataLoader for the training dataset.
    val_loader : `torch.utils.data.DataLoader`
        DataLoader for the validation dataset.
    patience : `int`, optional
        Number of epochs to wait without improvement in validation loss before
        triggering early stopping. Default is 10.

    Returns
    -------
    None

    Notes
    -------
    The function uses an early stopping strategy to save the model if no
    improvement in validation loss is observed for `patience` consecutive
    epochs. The model is saved with a timestamped filename in a subdirectory
    `best_model`.

    If `VISUALIZE_RECONSTRUCTION` is set to True in `config.py`, the function 
    also plots input vs. reconstructed images from the validation set after 
    training.
    """

    # Set model to training mode
    model.train()

    # Set current best_val_loss to inf and epochs_not_improved to 0 for early stopping strategy
    best_val_loss = float("inf")
    epochs_not_improved = 0

    # Run epochs
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        # For each batch:
        for x in train_loader:

            # Move batch to DEVICE, and reset gradients to zero
            x = x.to(DEVICE)
            optimizer.zero_grad()

            # Run a forward pass
            x_hat, mean, log_var = model(x)

            # Calculate the loss, backpropagate and do optimizer step
            loss = loss_function(x, x_hat, mean, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate average training loss
        avg_loss = total_loss / len(train_loader.dataset)

        # Calculate average validation loss
        val_loss = validate(model, val_loader)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_loss:.2f}, Validation Loss: {val_loss:.2f}, Epoch duration: {epoch_time:.2f}",
            flush=True,
        )

        # Early Stopping strategy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_not_improved = 0
            best_model = model
        else:
            epochs_not_improved += 1

        if epochs_not_improved >= patience:
            print("Early stopping triggered!")
            if best_model is not None:
                torch.save(
                    best_model, os.path.join(script_dir, f'best_model/best_full_model_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.pt')
                )
                print(
                    f"Best model saved as 'best_full_model_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.pt'."
                )
            break

    # If config.py defines VISUALIZE_RECONSTRUCTION = True
    if VISUALIZE_RECONSTRUCTION:
        # At last epoch, plot forward pass of input vs reconstruction
        visualize_reconstruction(best_model, val_loader)

    if epochs_not_improved < patience:
        print(f"Training completed after {EPOCHS} epochs.", flush=True)
        # Save the best model at the end of training if early stopping wasn't triggered
        if best_model is not None:
            torch.save(
                    best_model, os.path.join(script_dir, f'best_model/best_model_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.pt')
                )
            print(
                f"Best model saved as 'best_model_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.pt'."
            )


def validate(model, val_loader):
    """Performs a validation step to compute the average validation loss for
    the VAE model.

    Parameters
    ----------
    model : `torch.nn.Module`
        The VAE model to validate.
    val_loader : `torch.utils.data.DataLoader`
        DataLoader for the validation dataset.

    Returns
    -------
    avg_val_loss : `float`
        The average validation loss computed over the validation dataset.
    """

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(DEVICE)
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    return avg_val_loss


if __name__ == "__main__":
    # Load data
    train_loader, val_loader, test_loader = prepare_dataloaders()
    print("Dataloaders created", flush=True)

    # Initialize model and optimizer
    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(model, optimizer, train_loader, val_loader)
