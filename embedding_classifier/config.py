import torch

# Paths
AUMC_NPZ_DIR = 'AUMC_npzs'
TRAIN_DIR = 'embedding_classifier/aumc_data/datasets/train_set/'
VAL_DIR = 'embedding_classifier/aumc_data/datasets/val_set/'
TEST_DIR = 'embedding_classifier/aumc_data/datasets/test_set/'
NUMBER_OF_PATIENTS_TO_USE = None  # Set to None for all files

LABELS_XLSX_PATH = 'sleutel_EDLVHD_20240906.xlsx'

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1  # test split is whatever is left, so about 10%.

PRETRAINED_MODEL = 'pretrain_vae/best_model/best_full_model_2024-10-12_032831.pt' # Download from huggingface!

TRAIN_LABELS = 'embedding_classifier/aumc_data/datasets/train_set/train_labels.csv'
VAL_LABELS = 'embedding_classifier/aumc_data/datasets/val_set/val_labels.csv'
TEST_LABELS = 'embedding_classfiier/aumc_data/datasets/test_set/test_labels.csv'

# Training parameters
INPUT_SIZE = (272, 400)  # has to be divisible by 16! (ideally a power of 2)
BATCH_SIZE = 500
EPOCHS = 300
LEARNING_RATE = 1e-4
LATENT_DIM = 256
VISUALIZE_RECONSTRUCTION = True

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
