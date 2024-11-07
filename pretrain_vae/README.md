# A module to pretrain the VAE
This folder contains all the code to pretrain the Variational Autoencoder on ECG data saved in NPZ format.

Before running the code, ensure you have followed the instructions in the main folder (VALVE_AI) to install and activate the conda environment. 

### Splitting the NPZ dataset
First, the NPZ dataset should be split into a training, validation and test set. For this, ensure you have correctly defined the split ratios, the number of files to use, and the paths to the NPZ directory and to the place where the datasets should be saved in `config.py`. Next, from the current directory run:

```sh
python data/split_dataset.py
```

Now `data/datasets` should be filled with the directories `train_set`, `val_set` and `test_set` which should contain the number of NPZ files defined by the split ratios and the total number of files to use. 

### Training the VAE
Next, to train the model, ensure all variables are correctly defined in `config.py` and then, from the current directory, run:

```sh
python train.py
```

The model will start training now and print the training and validation loss each epoch. The best model will be saved in `best_model/` with a datetime stamp in the filename. If `VISUALIZE_RECONSTRUCTION` is set to `TRUE`, a reconstruction plot will be created and saved in `reconstructions/` also including a datetime stamp in the filename.

### Optional: Visualizing reconstruction
If you already have a pretrained model and want to plot some inputs alongside their reconstructed counterparts, you can use `visualize_saved_model.py`. Define the path to the pretrained model (BEST_MODEL) in config.py and change line 113 in `models/vae.py`, `epsilon = torch.randn_like(std).to(self.device)` , to `epsilon = torch.randn_like(std).to('cpu')` and run:

```sh
python visualize_saved_model.py
```

This will generate a reconstruction plot and save it in `reconstructions/`. It uses the function defined in `visualize.py` and thus this is where you can change any plot features.