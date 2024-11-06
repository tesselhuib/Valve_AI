# Valve_AI
This repository contains all code written for my 3 month internship at Amsterdam Center for Computational Cardiology. The report related to the internship can be found in the repository as well as: `internship_report.pdf`. The goal of the project was to detect valvular heart disease on ECG.

### Conda environment

To be able to use the code, the correct dependencies should be installed. For the correct dependencies, a yaml file called `conda_create.yml` is included in `./resources/`. A new environment called VALVE_AI can be built using the command:

```sh
# From the root of the repository
conda env create --file ./resources/conda_create.yml
```

### Repository structure
The repository is split into a separate folder for each part of the pipeline explained in the report. First, the data needs to be processed and converted to NPZ. This code can be found in `preprocessing/`. Then, the code for pretraining the variational autoencoder can be found in `pretrain_vae/`. Next, the code for classifying the embeddings generated by the VAE can be found in `embedding_classifier/` and lastly, the code for evaluation of the model can be found in `model_evaluation/`. Each directory contains a separate `README` with more information on how the code was used.

### Data and model availability
Due to privacy concerns, the data is not publically available. However, the pretrained models are. The pretrained VAE can be downloaded from (it cannot be uploaded to GitHub due to its size): https://huggingface.co/tesselhuib/ECG_encoder/tree/main. The trained XGBoost classification model can be found in `embedding_classifier/best_model/best_model.xgb`. 


