"""This module contains the VAE class that describes the variational
autoencoder. It consists of the Encoder class and the Decoder class also 
described in this module.

Notes
-----
The module requires the input size of the images, 'INPUT_SIZE', the latent
dimension 'LATENT_DIM', and the type of device used (cuda or cpu), 'DEVICE',
to be defined in a global 'config.py' file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import INPUT_SIZE, LATENT_DIM, DEVICE


class Encoder(nn.Module):
    """Encoder part of the VAE. Four convolutional layers deep.
    """

    def __init__(self, latent_dim=LATENT_DIM, input_size=INPUT_SIZE):
        super(Encoder, self).__init__()

        self.input_size = INPUT_SIZE

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)  # ((input-kernel+2*padding)/stride)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1) 

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)
            dummy_output = self.conv1(dummy_input)
            dummy_output = self.conv2(dummy_output)
            dummy_output = self.conv3(dummy_output)
            dummy_output = self.conv4(dummy_output)
            self.flattened_size = dummy_output.numel()
            self.output_shape = dummy_output.shape[1:]

        # Fully connected layers for mean and log variance
        self.fc_mean = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        self.input_size = x.shape
        # print("input size correct", self.input_size)
        x = self.LeakyReLU(self.conv1(x))
        x = self.LeakyReLU(self.conv2(x))
        x = self.LeakyReLU(self.conv3(x))
        x = self.LeakyReLU(self.conv4(x))

        # Flatten the output
        x = x.view(x.size(0), -1)
        # Get mean and log variance
        mean = self.fc_mean(x)
        log_var = self.fc_logvar(x)
        return mean, log_var


class Decoder(nn.Module):
    """Decoder of the VAE. """

    def __init__(self, latent_dim=LATENT_DIM, flattened_size=None, output_shape=None, target_size=INPUT_SIZE):
        super(Decoder, self).__init__()

        # Fully connected layer to map latent space to conv feature map
        self.fc = nn.Linear(latent_dim, flattened_size)
        self.output_shape = output_shape
        self.target_size = target_size

        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, z):
        # Map latent vector to feature map
        x = self.fc(z)
        x = x.view(x.size(0), *self.output_shape)
        x = self.LeakyReLU(self.deconv1(x))
        x = self.LeakyReLU(self.deconv2(x))
        x = self.LeakyReLU(self.deconv3(x))
        x_hat = torch.sigmoid(self.deconv4(x))

        return x_hat


class VAE(nn.Module):
    """The VAE, combining the encoder and decoder and including a reparameterization step.
    """

    def __init__(self, latent_dim=LATENT_DIM, device=DEVICE):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(
            latent_dim=latent_dim,
            flattened_size=self.encoder.flattened_size,
            output_shape=self.encoder.output_shape,
            target_size=self.encoder.input_size)
        self.device = DEVICE

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std).to(self.device)  # CHANGE TO 'cpu' FOR VISUALIZE_SAVED_MODEL.py, otherwise 'self.device'!
        z = mean + std*epsilon
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar
