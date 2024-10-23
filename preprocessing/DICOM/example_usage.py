""" Converting AUMC DICOMs to NPZ or PDF.
Deals with the directory structure as found in testData/DICOM, as this is how AUMC provides the DICOM data.

"""

import os
import sys

# To find ecgprocess, can also install ecgprocess as package, then remove this
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ecgProcess.constants import ProcessDicomNames as PDNames
from plot_ecgs import ECGDrawing
from process_dicoms import ECGDICOMReader, ECGDICOMTable

# Input directory containing the original DICOM structure
DIRECTORY = "testData/DICOM"

# Output directory where the NPZ files will be saved
TARGET_DIRECTORY = "testData/PDF"

# Filetype to convert to
TARGET_FILETYPE = "pdf"  # MUST BE 'npz' OR 'pdf'

# Define layout for plot
LAYOUT = [
    ["I", "V1"],
    ["II", "V2"],
    ["III", "V3"],
    ["aVR", "V4"],
    ["aVL", "V5"],
    ["aVF", "V6"],
]

UPDATE_KEYS = {
    "I (Einthoven)": "I",
    "Lead I (Einthoven)": "I",
    "Lead II": "II",
    "Lead III": "III",
    "Lead aVR": "aVR",
    "Lead aVL": "aVL",
    "Lead aVF": "aVF",
    "Lead V1": "V1",
    "Lead V2": "V2",
    "Lead V3": "V3",
    "Lead V4": "V4",
    "Lead V5": "V5",
    "Lead V6": "V6",
}

kwargs_drawing = {PDNames.PLOT_LAYOUT: LAYOUT}


if TARGET_FILETYPE == "npz":

    def convert_dicom(dicom_file, target_path):
        ecgdicomreader = ECGDICOMReader()
        ecgtable = ECGDICOMTable(
            ecgdicomreader=ecgdicomreader, path_list=[dicom_file]
        )()
        npz = ecgtable.write_numpy(
            ECGDrawing(update_keys=UPDATE_KEYS),
            target_path=target_path,
            kwargs_drawing=kwargs_drawing,
        )
        return npz

elif TARGET_FILETYPE == "pdf":

    def convert_dicom(dicom_file, target_path):
        ecgdicomreader = ECGDICOMReader()
        ecgtable = ECGDICOMTable(
            ecgdicomreader=ecgdicomreader, path_list=[dicom_file]
        )()
        pdf = ecgtable.write_pdf(
            ECGDrawing(update_keys=UPDATE_KEYS),
            target_path=target_path,
            kwargs_drawing=kwargs_drawing,
        )
        return pdf

else:
    raise KeyError(
        f"The following target file is not possible: {TARGET_FILETYPE}. Change to 'npz' or 'pdf'."
    )


def convert_and_save_dicom(study_dir, result_dir):
    for root, _, files in os.walk(study_dir):
        for file in files:
            if file.endswith(".dcm"):

                dicom_file_path = os.path.join(root, file)

                # Extract patient ID and measurement from the directory structure
                relative_path = os.path.relpath(dicom_file_path, study_dir)
                path_parts = relative_path.split(os.sep)
                patient_id = path_parts[0]

                # Create corresponding output dir structure in the result directory
                output_patient_dir = os.path.join(result_dir, patient_id)
                if not os.path.exists(output_patient_dir):
                    os.makedirs(output_patient_dir)

                # Generate target path for NPZ file in the new result directory
                target_path = os.path.join(output_patient_dir)

                # Convert DICOM and save NPZ using the generated target path
                convert_dicom(dicom_file_path, target_path)

                print(f"Converted {dicom_file_path} to {target_path}", flush=True)


# Call the function to start conversion and save NPZ files in the new structure
convert_and_save_dicom(DIRECTORY, TARGET_DIRECTORY)
