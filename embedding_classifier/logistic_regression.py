"""Script to perform a LASSO LogisticRegression on the embeddings as a
baseline score to test the performance of the XGBoost model against.

Notes
-----
Default values for the parameters  of LogisticRegressionCV are used.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

TRAIN_EMBEDDINGS = 'embeddings/train_embeddings_reduced.csv'
VAL_EMBEDDINGS = 'embeddings/val_embeddings_reduced.csv'


def main():
    # Load the training dataset from CSV
    train_df = pd.read_csv(TRAIN_EMBEDDINGS)
    X_train = train_df.iloc[:, 2:].values
    y_train = train_df['label'].values

    # Load the validation dataset from CSV
    val_df = pd.read_csv(VAL_EMBEDDINGS)
    X_val = val_df.iloc[:, 2:].values
    y_val = val_df['label'].values

    # Create a LogisticRegressionCV model
    model = LogisticRegressionCV(cv=3,
                                penalty='l1',
                                solver='saga',
                                scoring='roc_auc',
                                random_state=42,
                                max_iter=1000,
                                tol=0.001,
                                n_jobs=-1,
                                verbose=0)

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_val)
    roc_auc = roc_auc_score(y_val, y_pred)

    # Print the roc auc score
    print("ROC AUC:", roc_auc)


if __name__ == "__main__":
    main()
