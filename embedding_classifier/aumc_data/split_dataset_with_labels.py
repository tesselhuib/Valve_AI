""" Script to organize AUMC NPZ files into training, validation, and test
directories and include a label file for each of these directories.

Attributes
----------
AUMC_NPZ_DIR : `str`
    The directory containing the AUMC NPZ files to be organized.
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
import random
import shutil
import pandas as pd
import csv
from config import (
    AUMC_NPZ_DIR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    NUMBER_OF_PATIENTS_TO_USE,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    LABELS_XLSX_PATH,
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


def load_labels_from_excel(label_path):
    """Loads patient labels from the Excel file.

    Parameters
    ----------
    label_path : `str`
        Path to xlsx file with label info.

    Returns
    -------
    dict
        Dictionary with patient IDs and their respective labels.
    """
    df_labels = pd.read_excel(label_path)

    # Create a dictionary mapping patient IDs to labels
    return {
        int(row["Pseudo_id"]): 0 if row["IsControlePatient"] == "Ja" else 1
        for _, row in df_labels.iterrows()
    }


def group_npz_files_by_patient(npz_dir, label_dict):
    """Traverses NPZ_DIR and groups ECGs by patient ID.

    Parameters
    ----------
    npz_dir : `str`
        The path to the directory with NPZ files.
    label_dict : dict
        A dictionary mapping patient IDs to labels.

    Returns
    -------
    patient_files : dict
        A dictionary with patient IDs as key and a list of tuples as value.
        The tuple contains the path to the ECG file and its associated label.
    total_ecgs : `int`
        Total number of ECGs in the NPZ dir.
    """
    patient_files = {}
    total_ecgs = 0
    for root, dirs, files in os.walk(npz_dir):
        for file in files:
            if file.endswith(".npz"):
                patient_id = os.path.basename(root)
                try:
                    patient_id = int(patient_id)
                except ValueError:
                    print(f"Skipping directory {root}, patient ID not valid.")
                    continue
                if patient_id in label_dict:
                    if patient_id not in patient_files:
                        patient_files[patient_id] = []
                    patient_files[patient_id].append(
                        (os.path.join(root, file), label_dict[patient_id])
                    )
                    total_ecgs += 1
    return patient_files, total_ecgs


def add_patients_to_split(target_ecg_count, patient_ids, patient_files):
    """Assigns patients to a split while monitoring the number of ECGs.

    Parameters
    ----------
    target_ecg_count : `int`
        The number of ECGs wanted in the dataset.
    patient_ids : `list` [`str`]
        The list of patient IDs to consider for splitting.
    patient_files : dict
        A dictionary mapping patient IDs to their ECG files.

    Returns
    -------
    current_ecg_count : `int`
        Number of ECGs added to directory
    split_patients : `list` [`str`]
        List of assigned patients.
    """
    current_ecg_count = 0
    split_patients = []
    while current_ecg_count < target_ecg_count and patient_ids:
        patient_id = patient_ids.pop(0)
        patient_ecg_count = len(patient_files[patient_id])
        if current_ecg_count + patient_ecg_count <= target_ecg_count:
            split_patients.append(patient_id)
            current_ecg_count += patient_ecg_count
        else:
            split_patients.append(patient_id)
            current_ecg_count += patient_ecg_count
            break  # Stop adding more patients after this
    return current_ecg_count, split_patients


def copy_files(patient_list, target_dir, patient_files):
    """Copies the ECG files of each patient to the target directory.

    Parameters
    ----------
    patient_list : `list` [`str`]
        List of assigned patients.
    target_dir : `str`
        Target directory where ECG files should be copied.
    patient_files :  dict
        A dictionary with patient IDs as key and a list of tuples as value.
        The tuple contains the path to the ECG file and its associated label.
    """
    for patient_id in patient_list:
        for file, label in patient_files[patient_id]:
            patient_dir = os.path.join(target_dir, str(patient_id))
            os.makedirs(patient_dir, exist_ok=True)
            shutil.copy(file, patient_dir)


def save_labels(patient_list, label_file, patient_files):
    """Saves the labels of the files for each split in a CSV.

    Parameters
    ----------
    patient_list : `list` [`str`]
        List of patient IDs for which to save labels.
    label_file : `str`
        Full path to the label file to be created.
    patient_files : dict
        A dictionary mapping patient IDs to their ECG files and labels.
    """
    with open(label_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Pseudo_id", "ECG File", "Label"])
        for patient_id in patient_list:
            for file, label in patient_files[patient_id]:
                writer.writerow([patient_id, os.path.basename(file), label])


def main():
    # Ensure output directories exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Clear directories if they are not empty
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not is_empty(directory):
            print(f"Clearing contents of {directory}...")
            clear_directory(directory)

    # Load patient labels from the Excel file
    label_dict = load_labels_from_excel(LABELS_XLSX_PATH)

    # Group NPZ files by patient
    patient_files, total_ecgs = group_npz_files_by_patient(AUMC_NPZ_DIR, label_dict)
    print(f"Total number of ECGs: {total_ecgs}")

    # Shuffle patient IDs for random splitting
    patient_ids = list(patient_files.keys())

    # Shuffle the list of NPZs
    if NUMBER_OF_PATIENTS_TO_USE is not None:
        patient_ids = random.sample(patient_ids, NUMBER_OF_PATIENTS_TO_USE)
    else:
        random.shuffle(patient_ids)

    # Calculate total ECGs based on selected patients
    selected_ecgs = sum(len(patient_files[patient_id]) for patient_id in patient_ids)
    print(f"Total number of ECGs from selected patients: {selected_ecgs}")

    # Calculate the target number of ECGs for each split
    train_target = int(selected_ecgs * TRAIN_SPLIT)
    val_target = int(selected_ecgs * VAL_SPLIT)
    test_target = selected_ecgs - train_target - val_target
    print(f"Target ECGs: Train={train_target}, Validation={val_target}, Test={test_target}\n")

    test_patients = []

    train_ecg_count, train_patients = add_patients_to_split(
        train_target, patient_ids, patient_files
    )
    val_ecg_count, val_patients = add_patients_to_split(
        val_target, patient_ids, patient_files
    )
    test_patients.extend(patient_ids)  # Remaining patients go to the test set

    # Copy files and save labels for each split
    copy_files(train_patients, TRAIN_DIR, patient_files)
    copy_files(val_patients, VAL_DIR, patient_files)
    copy_files(test_patients, TEST_DIR, patient_files)

    save_labels(
        train_patients,
        os.path.join(TRAIN_DIR, "train_labels.csv"),
        patient_files,
    )
    save_labels(val_patients, os.path.join(VAL_DIR, "val_labels.csv"), patient_files)
    save_labels(
        test_patients,
        os.path.join(TEST_DIR, "test_labels.csv"),
        patient_files,
    )

    test_ecg_count = sum(len(patient_files[patient_id]) for patient_id in test_patients)

    # Print summary
    print(
        f"Train set: {len(train_patients)} patients, {train_ecg_count} ECGs copied to {TRAIN_DIR}\n"
    )
    print(
        f"Validation set: {len(val_patients)} patients, {val_ecg_count} ECGs copied to {VAL_DIR}\n"
    )
    print(
        f"Test set: {len(test_patients)} patients, {test_ecg_count} ECGs copied to {TEST_DIR}\n"
    )


# Entry point
if __name__ == "__main__":
    main()
