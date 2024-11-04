import torch

# Data paths
NPZ_DIR = 'ukbiobank_npz_without_movement_artefacts/'
TRAIN_DIR = 'VAE_npz/data/datasets/train_set/'
VAL_DIR = 'VAE_npz/data/datasets/val_set/'
TEST_DIR = 'VAE_npz/data/datasets/test_set/'

NUMBER_OF_FILES_TO_USE = 30000  # Set to None for all files

# Training parameters
INPUT_SIZE = (272, 400)  # has to be divisible by 16! (ideally a power of 2)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1  # test split is whatever integer number of files is left, so about 10%.
BATCH_SIZE = 500
EPOCHS = 300
LEARNING_RATE = 1e-4
LATENT_DIM = 256
VISUALIZE_RECONSTRUCTION = True

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path to best trained model
BEST_MODEL = "best_model/best_full_model_2024-10-12_032831.pt"
