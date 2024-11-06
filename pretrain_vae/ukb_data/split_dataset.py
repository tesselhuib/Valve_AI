"""
Script to organize UK Biobank NPZ files into training, validation, and test
directories.

Attributes
----------
UKB_NPZ_DIR : `str`
    The directory containing the UK Biobank NPZ files to be organized.
TRAIN_DIR : `str`
    The directory where the training NPZ files will be copied.
VAL_DIR : `str`
    The directory where the validation NPZ files will be copied.
TEST_DIR : `str`
    The directory where the test NPZ files will be copied.
TRAIN_SPLIT : `float`
    Proportion of the dataset to use for training.
VAL_SPLIT : `float`
    Proportion of the dataset to use for validation.
NUMBER_OF_FILES_TO_USE : `int`, optional
    The maximum number of NPZ files to use from the NPZ directory. If None,
    all files will be used.

Notes
-----
This script reads NPZ files from a specified directory, splits them into
training, validation, and test sets according to specified ratios, and copies
them to their respective output directories. It also clears any existing files
in these directories before copying new ones. All the attributes are expected
to be loaded in from a global 'config.py' file.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import random
import shutil
from config import (
    UKB_NPZ_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    NUMBER_OF_FILES_TO_USE,
)


def is_empty(directory):
    """Check if a directory is empty.

    Parameters
    ----------
    directory : `str`
        Path to the directory to check.

    Returns
    -------
    bool
        True if the directory is empty, False otherwise.
    """
    return not os.listdir(directory)


def clear_directory(directory):
    """Clear the contents of a directory.

    If the directory exists, it will be deleted and recreated.

    Parameters
    ----------
    directory : `str`
        Path to the directory to clear.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        os.makedirs(directory)


def copy_files(file_list, target_dir):
    """Copy a list of files to a target directory.

    Parameters
    ----------
    file_list : `list` [`str`]
        List of file paths to copy.
    target_dir : `str`
        Path to the target directory where files will be copied.
    """
    for file in file_list:
        shutil.copy(file, target_dir)


def main():
    """Main function to organize NPZ files into training, validation, and test
    directories.
    """

    # Create output directories if they don't exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Check and clear directories if they are not empty
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not is_empty(directory):
            print(f"Clearing contents of {directory}...")
            clear_directory(directory)

    # Get list of NPZ files
    npz_files = glob.glob(os.path.join(UKB_NPZ_DIR, "*.npz"))

    # Shuffle the list of NPZs
    if NUMBER_OF_FILES_TO_USE is not None:
        npz_files = random.sample(npz_files, NUMBER_OF_FILES_TO_USE)
    else:
        random.shuffle(npz_files)

    # Calculate the split indices
    total_files = len(npz_files)
    train_end = int(total_files * TRAIN_SPLIT)
    val_end = int(total_files * (TRAIN_SPLIT + VAL_SPLIT))

    # Split into train, validation, and test sets
    train_files = npz_files[:train_end]
    val_files = npz_files[train_end:val_end]
    test_files = npz_files[val_end:]

    # Copy files to their respective directories
    copy_files(train_files, TRAIN_DIR)
    copy_files(val_files, VAL_DIR)
    copy_files(test_files, TEST_DIR)

    print(f"Train set: {len(train_files)} NPZs copied to {TRAIN_DIR}")
    print(f"Validation set: {len(val_files)} NPZs copied to {VAL_DIR}")
    print(f"Test set: {len(test_files)} NPZs copied to {TEST_DIR}")


if __name__ == "__main__":
    main()
