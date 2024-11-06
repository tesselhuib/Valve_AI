import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.utils import resample

# Define paths to test embeddings and to the trained xgboost model
TEST_EMBEDDINGS = "embedding_classifier/embeddings/test_embeddings_reduced.csv"
BEST_XGB_MODEL = "embedding_classifier/best_model/best_model.xgb"


def load_data(fpath):
    """Loads data from a csv file

    Parameters
    ----------
    fpath : `str`
        File path to the csv file

    Returns
    -------
    X : pandas.Dataframe
        Dataframe with all the embeddings for each patient
    y : pandas.Dataframe
        Dataframe with all the labels for each patient
    """

    df = pd.read_csv(fpath)
    X = df.iloc[:, 2:].values
    y = df["label"].values

    return X, y


def compute_roc_ci_interval(y_test, y_pred, n_iterations=1000):
    """Computes the 95% confidence interval (CI) of the ROC-AUC value

    Parameters
    ----------
    y_test : pandas.Dataframe
        Dataframe consisting of the true label for each patient
    y_pred : pandas.Dataframe
        Dataframe consisting of the predicted probability (between 0 and 1)
        for VHD for each patient.
    n_iterations : `int`, optional
        Number of iterations to calculate the CI. Defaults to 1000.
    """
    bootstrapped_scores = []

    for _ in range(n_iterations):
        # Resample with replacement both y_test and y_pred
        indices = resample(range(len(y_test)), replace=True)
        # Check if both classes are in the resampled data
        if len(np.unique(y_test[indices])) < 2:
            continue  # Skip if only one class is present

        # Calculate the ROC AUC score for this bootstrap sample
        score = roc_auc_score(y_test[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    bootstrapped_scores = np.array(bootstrapped_scores)

    lower_bound = np.percentile(bootstrapped_scores, 2.5)
    upper_bound = np.percentile(bootstrapped_scores, 97.5)

    print(f"ROC-AUC 95% Confidence Interval: ({lower_bound:.4f}, {upper_bound:.4f})")


def create_roc_precision_recall_curve(fpr, tpr, roc_auc, y_test, y_pred):
    """Creates a plot with on the left side the ROC curve and on the right
    side the precision recall curve.

    Parameters
    ----------
    fpr : `ndarray`
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].
    tpr : `ndarray`
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].
    roc_auc : `float`
        Area under the curve for the ROC curve.
    y_test : pandas.Dataframe
        Dataframe consisting of the true label for each patient
    y_pred : pandas.Dataframe
        Dataframe consisting of the predicted probability (between 0 and 1)
        for VHD for each patient.
    """

    # Plot ROC curve
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(
        fpr, tpr, color="blue", lw=2, label="ROC curve (AUC = {:.2f})".format(roc_auc)
    )
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("ROC Curve")

    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title("Precision-Recall Curve")
    plt.savefig("roc_and_precision_curves.pdf")


def create_predicted_probabilities_density_plot(y_test, y_pred):
    """Creates an overlapping density plot of predicted VHD probabilites for
    the patients with true label VHD and with true label Non-VHD.

    Parameters
    ----------
    y_test : pandas.Dataframe
        Dataframe consisting of the true label for each patient
    y_pred : pandas.Dataframe
        Dataframe consisting of the predicted probability (between 0 and 1)
        for VHD for each patient.
    """

    # # Separate predictions based on y_test values
    y_pred_1 = y_pred[y_test == 1]  # Predictions where the true label is 1
    y_pred_0 = y_pred[y_test == 0]  # Predictions where the true label is 0

    # Create overlapping density plots
    plt.figure(figsize=(10, 5))
    plt.hist(
        y_pred_1,
        bins=30,
        density=True,
        alpha=0.5,
        color="red",
        edgecolor="black",
        label="VHD",
    )
    plt.hist(
        y_pred_0,
        bins=30,
        density=True,
        alpha=0.5,
        color="blue",
        edgecolor="black",
        label="Non-VHD",
    )

    # Labels and title
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend()

    # Save the plot
    plt.savefig("overlapping_predicted_probs_density.pdf")

    plt.figure(figsize=(10, 5))
    plt.hist(y_pred, bins=30, alpha=0.7, color="blue", edgecolor="black")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Probabilities")
    plt.grid()
    plt.savefig("predicted_probs.pdf")


def create_performance_table(y_test, y_pred):
    """Prints the performance table for the XGBoost classification model

    Parameters
    ----------
    y_test : pandas.Dataframe
        Dataframe consisting of the true label for each patient
    y_pred : pandas.Dataframe
        Dataframe consisting of the predicted probability (between 0 and 1)
        for VHD for each patient.

    Notes
    -----
    The table includes sensitivity, specificity, positive predicitive rate
    and negative predictive rate for 5 different thresholds.
    """
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    # Initialize an empty list to store results for each threshold
    results = []

    for threshold in thresholds:
        # Classify predictions based on the current threshold
        y_pred_class = (y_pred >= threshold).astype(int)

        # Calculate TP, FP, TN, FN
        TP = np.sum((y_pred_class == 1) & (y_test == 1))
        FP = np.sum((y_pred_class == 1) & (y_test == 0))
        TN = np.sum((y_pred_class == 0) & (y_test == 0))
        FN = np.sum((y_pred_class == 0) & (y_test == 1))

        sens = TP / (TP + FN)  # sensitivity
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        spec = TN / (FP + TN)  # specificity
        # Append the results as a dictionary
        results.append(
            {
                "Threshold": threshold,
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "sens": sens,
                "spec": spec,
                "PPV": PPV,
                "NPV": NPV,
            }
        )

    # Convert the results to a DataFrame for easy viewing
    results_df = pd.DataFrame(results)

    # Display the table
    print(results_df)


def main():

    # Load the testing dataset from CSV
    X_test, y_test = load_data(TEST_EMBEDDINGS)

    dtest = xgb.DMatrix(X_test, label=y_test)

    xgb_model = xgb.Booster()
    xgb_model.load_model(BEST_XGB_MODEL)

    # Make predictions on the test set
    y_pred = xgb_model.predict(dtest)

    # Compute ROC curve and ROC AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Compute ROC-AUC 95% CI
    compute_roc_ci_interval(y_test, y_pred)

    # Create density plot of predicted probabilities split per true label
    create_predicted_probabilities_density_plot(y_test, y_pred)

    # Create combination plot of ROC curve and Precision-Recall curve
    create_roc_precision_recall_curve(fpr, tpr, roc_auc, y_test, y_pred)

    # Create table with XGBoost performance metrics
    create_performance_table(y_test, y_pred)


if __name__ == "__main__":
    main()
