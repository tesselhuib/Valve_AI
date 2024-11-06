import torch

# Data paths - CHANGE ACCORDING TO YOUR FILE PATHS
UKB_NPZ_DIR = 'ukbiobank_npzs/'
TRAIN_DIR = 'pretrain_vae/ukb_data/datasets/train_set/'
VAL_DIR = 'pretrain_vae/ukb_data/datasets/val_set/'
TEST_DIR = 'pretrain_vae/ukb_data/datasets/test_set/'

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
BEST_MODEL = "pretrain_vae/best_model/best_full_model_2024-10-12_032831.pt"
