# Preprocessing data

This folder contains the code to process ECG data in either .xml format or .dcm format to .npz in order to process it in a variational autoencoder. The main folders are `DICOM/` and `XML/`, which both use code from `ecgProcess/` to be able to convert the filetypes to NPZ.

### DICOM to NPZ or PDF
The `DICOM/` directory contains the code to convert DICOM to PDF or to NPZ. For this internship project data from Amsterdam UMC was used, which contains ECGs saved in DICOM format. To process these, conversion to npz was needed. The code in this folder is heavily based on https://gitlab.com/SchmidtAF/ECGProcess , with some modifications to be able to convert to NPZ as well. 

The main modification is the addition of a write_numpy() function in the ECGDICOMTable() class. Next to that, two additions were made. One, was the inclusion of the check_all_ecg_leads_threshold() function in the ECGDrawing() class to remove ecgs of which one or more leads had more than 10% of samples with an absolute value > 1.5 mV. These ECGs were suspected of motion artefacts and therefore not included in the pretraining of the VAE and not converted. The second addition is that of the check whether a Median Waveform was present before converting to pdf or npz, if it was not present the ECG recording failed and therefore these ECGs were not converted either.

The way the code was used for this internship project can be found in `DICOM/example_usage.py`.

### XML to NPZ or PDF
The `XML/` directory contains the code to convert XML to PDF or to NPZ. For the pretraining in this internship project the UK Biobank database was used, which contains ECGs saved in XML format. To process these in a variational autoencoder, conversion to npz was needed. The code in this folder is also heavily based on https://gitlab.com/SchmidtAF/ECGProcess , with some modifications to handle XML instead of DICOM.

The main modification is in the __call__() function of the ECGXMLReader() class to process XML instead of DICOM. Here, we also included the subtraction of the mean for each ECG lead trace to ensure a zero mean baseline for each ECG lead. Next to that the same modifications as noted above for the `DICOM/` folder were included as well.

The way the code was used for this internship project can be found in `XML/example_usage.py`.

### Exclude ECGs recorded after a valvular intervention
As explained in the report, DICOMs of ECGs recorded after valvular intervention were excluded. The code to find these ECGs can be found in `exclude_interventionECG.py`. This creates `excluded_npz_paths.txt`, which contains the paths to the NPZs that should be excluded due to this reason. If wanted, these can be deleted from the NPZ directory.