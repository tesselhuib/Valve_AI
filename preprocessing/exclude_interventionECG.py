"""Script to exclude ECGs recorded after a valvular intervention.
"""
import os
import pandas as pd

# Define paths
NPZ_DIR = 'AUMC_NPZs'
DICOM_DIR = 'AUMC_DICOMs'
INTERVENTION_FILE = 'Intervention_data.xlsx'
OUTPUT_FILE = 'excluded_npz_paths.txt'

# Load intervention dates from Excel
intervention_df = pd.read_excel(INTERVENTION_FILE)
# Assume Excel has columns 'patient_id' and 'intervention_date' formatted as YYYY-MM-DD
intervention_df['interv_datum1'] = pd.to_datetime(intervention_df['interv_datum1'], errors='coerce')

# Initialize a list to store paths to exclude
excluded_npz_paths = []

# Iterate over each patient folder in the NPZ directory
for patient_id in os.listdir(NPZ_DIR):
    patient_npz_path = os.path.join(NPZ_DIR, patient_id)
    if os.path.isdir(patient_npz_path):
        # Check if this patient_id exists in the intervention dates DataFrame
        patient_data = intervention_df[intervention_df['Pseudo_id'] == int(patient_id)]
        
        if patient_data.empty or pd.isna(patient_data['interv_datum1'].values[0]):
            print("No intervention date for this patient_id, skip", patient_id)
            continue
        
        intervention_date = patient_data['interv_datum1'].values[0]
        
        # Get the corresponding DICOM directory for this patient
        patient_dicom_path = os.path.join(DICOM_DIR, patient_id)
        if not os.path.exists(patient_dicom_path):
            print("No corresponding DICOM folder, skip", patient_id)
            continue

        # Dictionary to hold the DICOM date associated with each npz file
        dicom_date_mapping = {}

        # Iterate over each subdirectory in the DICOM folder for dates
        for date_folder in os.listdir(patient_dicom_path):
            # Extract the date part before the "_"
            date_str = date_folder.split('_')[0]
            try:
                ecg_date = pd.to_datetime(date_str, format='%Y%m%d')
            except ValueError:
                # If date format is incorrect, skip
                print("ValueError")
                continue

            # Map each npz file in this date folder to the ECG date
            ecg_path = os.path.join(patient_dicom_path, date_folder)
            for root, dirs, files in os.walk(ecg_path):
                # Look for the subdirectory with DICOM files
                if files:
                    for dicom_file in files:
                        # Create a mapping based on the filename, assuming npz and dicom filenames align
                        npz_filename = dicom_file.replace('.dcm', '.npz')
                        dicom_date_mapping[npz_filename] = ecg_date
                    break  # Stop after finding the first directory with files

        # Check npz files for dates and add to exclusion if date is after intervention
        for npz_file in os.listdir(patient_npz_path):
            npz_date = dicom_date_mapping.get(npz_file)
            if npz_date and npz_date > intervention_date:
                excluded_npz_paths.append(os.path.join(patient_npz_path, npz_file))

# Write excluded paths to output file
with open(OUTPUT_FILE, 'w') as f:
    for path in excluded_npz_paths:
        f.write(f"{path}\n")

print(f"Excluded NPZ paths saved to {OUTPUT_FILE}")
