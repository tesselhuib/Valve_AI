"""Example usage of how to convert ECGs saved in xml to pdf or npz."""

import os
import sys

# To find ecgProcess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ecgProcess.constants import ProcessDicomNames as PDNames
from plot_ecgs import ECGDrawing
from process_xmls import ECGXMLReader, ECGXMLTable

# Set directories where xmls can be found and where to save the converted xmls
DIRECTORY = "testData/XML"
TARGET_DIRECTORY = "testData/PDF"
TARGET_FILETYPE = "pdf"  # MUST BE 'npz' OR 'pdf'

# Choose lay-out for plot
LAYOUT = [
    ["I", "V1"],
    ["II", "V2"],
    ["III", "V3"],
    ["aVR", "V4"],
    ["aVL", "V5"],
    ["aVF", "V6"],
]

# Add layout argument to kwargs_drawing
kwargs_drawing = {PDNames.PLOT_LAYOUT: LAYOUT}

# Create list of filepaths to process
filepaths = [os.path.join(DIRECTORY, filename) for filename in os.listdir(DIRECTORY)]
filepaths.sort()

# Process the xmls and write them to npz or pdf using a ECGXMLTable instance
ecgxmlreader = ECGXMLReader()
ecgtable = ECGXMLTable(ecgxmlreader=ecgxmlreader, path_list=filepaths)()

if TARGET_FILETYPE == "npz":
    npzs = ecgtable.write_numpy(
        ECGDrawing(), target_path=TARGET_DIRECTORY, kwargs_drawing=kwargs_drawing
    )
elif TARGET_FILETYPE == "pdf":
    pdfs = ecgtable.write_pdf(
        ECGDrawing(), target_path=TARGET_DIRECTORY, kwargs_drawing=kwargs_drawing
    )
else:
    raise KeyError(
        f"The following target file is not possible: {TARGET_FILETYPE}. Change to 'npz' or 'pdf'."
    )
