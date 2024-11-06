""" Script to train a XGBoost classification model.

Notes
-----
The hyperparameters have to be defined at the top of the script. A
hyperparameter search was used to find the current parameters. After
training, the best model will be saved to 'best_model/best_model.xgb'.
"""

import pandas as pd
import xgboost as xgb  # Import XGBoost

# Define file paths to embeddings
TRAIN_EMBEDDINGS = 'embeddings/train_embeddings_reduced.csv'
VAL_EMBEDDINGS = 'embeddings/val_embeddings_reduced.csv'

# Define hyperparameters
TRAIN_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 10,
    'min_child_weight': 20,
    'learning_rate': 0.03,
    'subsample': 0.91,
    'random_state': 42,
}


def main():
    # Load the training dataset from CSV
    train_df = pd.read_csv(TRAIN_EMBEDDINGS)
    X_train = train_df.iloc[:, 2:].values
    y_train = train_df['label'].values

    print("Training embeddings shape:", X_train.shape)

    # Load the validation dataset from CSV
    val_df = pd.read_csv(VAL_EMBEDDINGS)
    X_val = val_df.iloc[:, 2:].values
    y_val = val_df['label'].values

    print("Validation embeddings shape:", X_val.shape)

    # Convert the data into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set parameters for training
    params = TRAIN_PARAMS

    evals_result = {}

    # Train the XGBoost model with validation
    evals = [(dtrain, 'train'), (dval, 'val')]
    xgb_model = xgb.train(params, dtrain, num_boost_round=424, evals=evals,
                         evals_result=evals_result, early_stopping_rounds=10)

    # Save the best model
    xgb_model.save_model('best_model/best_model.xgb')
    print(f"Best model saved at iteration {xgb_model.best_iteration} to 'best_model/best_model.xgb'")


if __name__ == "__main__":
    main()
