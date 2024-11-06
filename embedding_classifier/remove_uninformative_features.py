"""Script to remove uninformative features from embeddings.

Notes
-----
Both features that are highly correlated with other features as well as
features with a low variance are excluded. Thresholds have to be defined
at the beginning of the script. The reduced sets will be saved in the
'embeddings/' folder as 'train_embeddings_reduced.csv',
'val_embeddings_reduced.csv' and 'test_embeddings_reduced.csv'.
"""

import pandas as pd
import numpy as np

# Define thresholds
CORRELATION_THRESHOLD = 0.8  # Features with correlation coefficients above this threshold will be excluded.
VARIATION_THRESHOLD = 0.05  # Features with a variance below this threshold will be excluded.

# File paths to embeddings
TRAIN_EMBEDDINGS = "embeddings/train_embeddings.csv"
VAL_EMBEDDINGS = "embeddings/val_embeddings.csv"
TEST_EMBEDDINGS = "embeddings/test_embeddings.csv"


def find_correlated_features(combined_df, th):
    """Finds the highly correlated features based on threshold 'th'.

    Parameters
    ----------
    combined_df : pandas.Dataframe
        Concatenated dataframe including all latent dimensions for the train,
        validation and test data.
    th : `float`
        The threshold above which a correlation coefficient of two features is
        considered high and therefore leads to exclusion of one of the two
        features.

    Returns
    -------
    correlated_features : set of `str`
        Set of feature names that will be excluded due to their high
        correlation coefficient with another feature.
    """

    # Calculate the correlation matrix for the combined embeddings
    corr_matrix = combined_df.corr(method="spearman")

    # Find highly correlated features (with threshold)
    highly_correlated = np.where(np.abs(corr_matrix) > th)
    highly_correlated = [
        (corr_matrix.index[x], corr_matrix.columns[y])
        for x, y in zip(*highly_correlated)
        if x != y and x < y
    ]

    # Print highly correlated pairs
    print(f"Highly correlated features (correlation > {th}):")
    for feature_pair in highly_correlated:
        print(f"{feature_pair[0]} and {feature_pair[1]}")

    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > th:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)

    return correlated_features


def find_lowvariance_features(combined_df, th):
    """Finds the features with a low variance based on threshold 'th'.

    Parameters
    ----------
    combined_df : pandas.Dataframe
        Concatenated dataframe including all latent dimensions for the train,
        validation and test data.
    th : `float`
        The threshold below which the variance of a feature is considered low
        and therefore leads to its exclusion.

    Returns
    -------
    low_variance_features : set of `str`
        Set of feature names that will be excluded due to their low variance.
    """

    # Calculate the variance for each feature
    variances = combined_df.var()

    # Identify features to drop based on variance
    low_variance_features = variances[variances < th].index.tolist()
    print(f"Low variance features: {low_variance_features}")

    return low_variance_features


def remove_features(
    train_df, val_df, test_df, correlated_features, low_variance_features
):
    """Removes the highly correlated and low variance features from the
    original datasets.

    Parameters
    ----------
    train_df : pandas.Dataframe
        Dataframe of all latent dimensions for the train set.
    val_df : pandas.Dataframe
        Dataframe of all latent dimensions for the validation set.
    test_df : pandas.Dataframe
        Dataframe of all latent dimensions for the test set.
    correlated_features : set of `str`
        Set of feature names that will be excluded due to their high
        correlation coefficient with another feature.
    low_variance_features: set of `str`
        Set of feature names that will be excluded due to their low variance.

    Returns
    -------
    train_embeddings_final : pandas.Dataframe
        Dataframe of remaining latent dimensions after exclusion for the train set.
    val_embeddings_final : pandas.Dataframe
        Dataframe of remaining latent dimensions after exclusion for the validation set.
    test_embeddings_final : pandas.Dataframe
        Dataframe of remaining latent dimensions after exclusion for the test set.
    """

    # Drop the features from train, validation, and test sets
    train_embeddings_reduced = train_df.drop(columns=correlated_features)
    val_embeddings_reduced = val_df.drop(columns=correlated_features)
    test_embeddings_reduced = test_df.drop(columns=correlated_features)

    train_embeddings_final = train_embeddings_reduced.drop(
        columns=low_variance_features
    )
    val_embeddings_final = val_embeddings_reduced.drop(columns=low_variance_features)
    test_embeddings_final = test_embeddings_reduced.drop(columns=low_variance_features)

    return train_embeddings_final, val_embeddings_final, test_embeddings_final


def main():
    # Load CSV files
    train_df = pd.read_csv(TRAIN_EMBEDDINGS)
    val_df = pd.read_csv(VAL_EMBEDDINGS)
    test_df = pd.read_csv(TEST_EMBEDDINGS)

    # Embeddings are from the 3rd column onwards (1st is filepath, second is label)
    train_embeddings = train_df.iloc[:, 2:]
    val_embeddings = val_df.iloc[:, 2:]
    test_embeddings = test_df.iloc[:, 2:]

    # Combine the data
    combined_df = pd.concat(
        [train_embeddings, val_embeddings, test_embeddings], ignore_index=True
    )

    # Find correlated and low variance features
    correlated_features = find_correlated_features(combined_df, CORRELATION_THRESHOLD)
    low_variance_features = find_lowvariance_features(combined_df, VARIATION_THRESHOLD)

    # Remove correlated and low variance features
    train_embed, val_embed, test_embed = remove_features(
        train_df, val_df, test_df, correlated_features, low_variance_features
    )

    print(f"Final training embeddings shape: {train_embed.shape}")
    print(f"Final validation embeddings shape: {val_embed.shape}")
    print(f"Final test embeddings shape: {test_embed.shape}")

    # Save the reduced datasets
    train_embed.to_csv("embeddings/train_embeddings_reduced.csv", index=False)
    val_embed.to_csv("embeddings/val_embeddings_reduced.csv", index=False)
    test_embed.to_csv("embeddings/test_embeddings_reduced.csv", index=False)


if __name__ == "__main__":
    main()
