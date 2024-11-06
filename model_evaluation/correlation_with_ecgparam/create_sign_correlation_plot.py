import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# Define paths to ECG parameters and to embeddings
ECG_PARAM_CSV = "dicom_ecgparams.csv"
EMBEDDINGS = "embedding_classifier/embeddings/train_embeddings.csv"


def get_spearman_matrix(merged_df, ecg_params, latent_columns):
    """Function to generate Spearman correlation matrix

    Parameters
    ----------
    merged_df : pandas.Dataframe
        Merged dataframe with ecg parameters and latent embeddings based on
        file name
    ecg_params : `list` of `str`
        List of the names of the ECG Parameters to include
    latent_columns : `list` of `str`
        List of the names of the latent embedding columns to include

    Returns
    -------
    correlation_df : pandas.Dataframe
        Dataframe containing the correlation coefficients for the ECG
        parameters with the latent embeddings
    p_value_matrix : numpy.Array
        NumPy array with the p values for every correlation.
    """
    # Initialize matrices to store correlation coefficients and p-values
    correlation_matrix = np.zeros((len(ecg_params), len(latent_columns)))
    p_value_matrix = np.zeros((len(ecg_params), len(latent_columns)))

    # Compute Spearman correlation and p-values for each pair
    for i, ecg_param in enumerate(ecg_params):
        for j, latent_col in enumerate(latent_columns):
            cleaned_df = merged_df[[ecg_param, latent_col]].dropna()
            corr, p_val = spearmanr(cleaned_df[ecg_param], cleaned_df[latent_col])
            correlation_matrix[i, j] = corr
            p_value_matrix[i, j] = p_val
    # Convert matrix to DataFrame for easier handling
    correlation_df = pd.DataFrame(
        correlation_matrix, index=ecg_params, columns=latent_columns
    )

    return correlation_df, p_value_matrix


def create_spearman_heatmap(correlation_df, significant_corrected_mask):
    """Function to create heatmap plot of the correlation coefficients.

    Parameters
    ----------
    correlation_df : pandas.Dataframe
        Dataframe containing the correlation coefficients for the ECG
        parameters with the latent embeddings
    significant_corrected_mask : pandas.Dataframe
        Dataframe of same size as correlation_df, with ones for the significant
        correlation coefficients and zeros for the insignificant correlation
        coefficients.

    Notes
    -----
    The heatmap only shows the significant correlation coefficients.
    """

    plt.figure(figsize=(24, 12))
    sns.heatmap(
        correlation_df,
        cmap="coolwarm",
        annot=False,
        fmt=".2f",
        vmin=-0.3,
        vmax=0.3,
        mask=~significant_corrected_mask,
    )
    plt.yticks(fontsize=22, rotation="horizontal")

    # # Set x-ticks and labels
    plt.xticks(ticks=[], labels=[])
    plt.xlabel("Latent embeddings", fontsize=22)
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.set_ylabel("Spearman correlation coefficient", size=22)
    plt.savefig("significant_spearman_correlations.pdf", bbox_inches="tight")


def main():
    # Load the ECG parameters & embeddings
    ecg_df = pd.read_csv(ECG_PARAM_CSV)
    latent_df = pd.read_csv(EMBEDDINGS)

    # Extract the file names (without extensions) from the file paths
    ecg_df["file_name"] = ecg_df["Filepath"].apply(
        lambda x: x.split("/")[-1].split(".dcm")[0]
    )
    latent_df["file_name"] = latent_df["file_path"].apply(
        lambda x: x.split("/")[-1].split(".npz")[0]
    )

    # Merge both DataFrames on the 'file_name' column
    merged_df = pd.merge(ecg_df, latent_df, on="file_name")

    # Select the ECG parameters and latent dimensions columns
    ecg_params = [
        "PR Interval",
        "QRS Duration",
        "QTc Interval",
        "RR Interval",
        "QT Interval",
    ]
    latent_columns = [f"latent_{i}" for i in range(256)]

    correlation_df, p_value_matrix = get_spearman_matrix(
        merged_df, ecg_params, latent_columns
    )

    # Apply significance threshold (e.g., p < 0.05)
    significance_level = 0.05

    # Apply multiple hypothesis correction using FDR (False Discovery Rate)
    reject, pvals_corrected, _, _ = multipletests(
        p_value_matrix.flatten(), alpha=significance_level, method="fdr_bh"
    )
    p_value_corrected_df = pd.DataFrame(
        pvals_corrected.reshape(p_value_matrix.shape),
        index=ecg_params,
        columns=latent_columns,
    )
    significant_corrected_mask = p_value_corrected_df < significance_level

    create_spearman_heatmap(correlation_df, significant_corrected_mask)
