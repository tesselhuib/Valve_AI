# XGBoost classification model on ECG embeddings

This folder contains the code to generate ECG embeddings using a pretrained variational autoencoder and to then train a XGBoost Classification model on these embeddings and their labels. Ensure you have followed the steps in the README of the main folder (Valve_AI) to install and activate the conda environment.

### 1. Split AUMC data
To generate the embeddings, first the preprocessed AUMC data (see 'preprocessing/') must be split into a train, validation and test set. This can be done using the script 'split_dataset_and_labels.py' in 'aumc_data/'. First change the AUMC_NPZ_DIR variable in 'config.py' to the path to the npz directory of the AUMC data and the LABELS_XLSX variable to the path to the Excel file that contains the labels for these patients. The Excel file has to have the columns "Pseudo_ID" and "IsControlePatient" which contains "Ja" for controls and "Nee" for cases. Also ensure that TRAIN_DIR, VAL_DIR and TEST_DIR point to the locations where you want the embeddings to be saved and define the split ratios 'TRAIN_SPLIT' and 'VAL_SPLIT'. Then Cd into the aumc_data folder and run: 

```sh
python split_dataset_with_labels.py
```

This will generate three folders within 'aumc_data/datasets', train_set, val_set and test_set, which will contain subdirectories of the patients included in those groups. Each subdirectory will have the pseudo ID of the patient as name and the ECGs of those patients as .npz files included in the subdirectory. The train_set, val_set and test_set directories will each also contain a file called 'train_labels.csv', 'val_labels.csv' and 'test_labels.csv' respectively, which contain the labels (0 for controls, 1 for cases) for each patient included in the set.

### 2. Generate embeddings
Next, the embeddings can be generated by performing a forward pass on the AUMC data through the pretrained VAE encoder. This is done in 'get_embeddings.py'. Ensure the attribuetes, as listed at the beginning of the script, are properly defined in 'config.py'. Ensure you have access to a GPU as the model is too heavy for a CPU. Then, in this directory, run:

```sh
python get_embeddings.py
```

The embeddings will be generated and saved in the 'embeddings/' folder as 'train_embeddings.csv', 'val_embeddings.csv' and 'test_embeddings.csv' respectively.

### 3. Remove uninformative features
After the embeddings are generated, we want to remove uninformative features. This consists of features that highly correlate with other features and features that show low variance over all samples. Their exclusion can be done by running the 'remove_uninformative_features.py' script. Within this script, the thresholds for exclusion can be defined, as well as the path to the embeddings. From the current directory, run:

```sh
python remove_uninformative_features.py
```

The uninformative features will be removed and the informative features will be saved in the 'embeddings/' folder as 'train_embeddings_reduced.csv', 'val_embeddings_reduced.csv' and 'test_embeddings_reduced.csv' respectively

### 4. Hyperparameter search
Now with the informative features, we can perform a hyperparameter search using Optuna. The script for this can be found in 'hyperparam_search.py'. In the script the ranges for the hyperparameter search are defined. From the current directory, run:

```sh
python hyperparam_search.py
```

The best hyperparameters will be saved as 'best_parameters.txt' for later reference.

### 5. Train XGBoost model
Now with the best hyperparameters, the xgboost model can be trained and the resulting model saved. This is done by running 'train_xgboost.py'. The hyperparameters can be altered within this file. From the current directory, run:

```sh
python train_xgboost.py
```

The XGBoost classification model will be trained and the best model will be saved in 'best_model/' as 'best_model.xgb'. 

For evaluation of the model, go to the 'model_evaluation/' folder.