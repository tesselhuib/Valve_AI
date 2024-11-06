"""Script to perform a hyperparameter search for the XGBoost classification
model.

Notes
-----
A hyperparameter search is done using Optuna. The hyperparameters are evaluated
based on the ROC-AUC score they achieve on the validation set.

Set the path to the training and validation embeddings at the start. Suggest
parameter ranges within the objective() function.

The best parameters will be saved as 'best_parameters.txt'.

"""
import pandas as pd
import xgboost as xgb
import optuna
import time
import plotly.io as pio
import optuna.visualization as vis
from sklearn.metrics import roc_auc_score

TRAIN_EMBEDDINGS = 'embeddings/train_emeddings_reduced.csv'
VAL_EMBEDDINGS = 'embeddings/val_embeddings_reduced.csv'

# Load the train dataset
train_df = pd.read_csv(TRAIN_EMBEDDINGS)
X_train = train_df.iloc[:, 2:].values
y_train = train_df['label'].values

print("Training embeddings shape:", X_train.shape)

# Load the validation dataset
val_df = pd.read_csv(VAL_EMBEDDINGS)
X_val = val_df.iloc[:, 2:].values
y_val = val_df['label'].values

print("Validation embeddings shape:", X_val.shape)

# Convert the data into DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)


# Define objective function for Optuna
def objective(trial):

    trial_start_time = time.time()

    # Suggest parameter ranges
    max_depth = trial.suggest_int('max_depth', 3, 11)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.15)
    n_estimators = trial.suggest_int('n_estimators', 250, 450)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'min_child_weight': 20,
        'random_state': 42,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
    }

    # Train the XGBoost model with validation
    evals = [(dtrain, 'train'), (dval, 'val')] 
    model = xgb.train(params, dtrain, num_boost_round=n_estimators, evals=evals,
                      early_stopping_rounds=10, verbose_eval=0)

    # Predict on validation set
    y_val_pred = model.predict(dval)

    # Calculate ROC-AUC score
    val_auc = roc_auc_score(y_val, y_val_pred)

    trial_end_time = time.time()
    trial_elapsed_time = trial_end_time - trial_start_time
    print(f"Trial finished: ROC AUC = {val_auc:.4f}, Time taken = {trial_elapsed_time:.2f}", flush=True)

    return val_auc


# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, n_jobs=-1)

# Get the best parameters and score
best_params = study.best_params
best_score = study.best_value

print(f"Best parameters found: {best_params}")
print(f"Best ROC AUC score: {best_score:.4f}")

with open('best_parameters.txt', 'w') as f:
    for param, value in best_params.items():
        f.write(f"{param}: {value}\n")

# Optional: plots to visualize optimization history and hyperparameter importance

# # Plot optimization history
# fig = vis.plot_optimization_history(study)
# pio.write_html(fig, file="/home/thuibregtsen/VAE+ResNET/interv_excl/optimization_history.html", auto_open=False)

# # Plot hyperparam importance
# fig_param_importance = vis.plot_param_importances(study)
# pio.write_html(fig_param_importance, file="/home/thuibregtsen/VAE+ResNET/interv_excl/param_importance.html", auto_open=False)