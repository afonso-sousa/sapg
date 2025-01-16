# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize


# Function to calculate length bins and mean ibleu for each bin
def calculate_mean_metric(df, metric="ibleu"):
    # drop last row which has mean values
    df = df[:-1]

    # Calculate source length
    df["source_length"] = df["source"].apply(lambda x: len(word_tokenize(x)))

    # Define bins and labels
    bins = [0, 5, 10, 15, 20, np.inf]
    labels = ["[0, 5]", "[5, 10]", "[10, 15]", "[15, 20]", "[20+]"]

    # Bin the source lengths
    df["length_bin"] = pd.cut(
        df["source_length"], bins=bins, labels=labels, right=False
    )

    # Calculate the mean ibleu for each bin
    mean_ibleu_per_bin = df.groupby("length_bin")[metric].mean()

    return mean_ibleu_per_bin


# Function to plot the graph for both datasets
def plot_mean_metric(
    mean_metric_1,
    mean_metric_2,
    label1,
    label2,
    x_label="source length",
    y_label="mean pair-wise iBLEU",
):
    # Create a line plot for the mean ibleu values per length bin for both files
    plt.figure(figsize=(10, 6))

    # Plot the first file's ibleu means
    mean_metric_1.plot(kind="line", marker="o", label=label1, color="orange")

    # Plot the second file's ibleu means
    mean_metric_2.plot(kind="line", marker="o", label=label2, color="green")

    # Set plot labels and title
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    # Show the plot
    plt.tight_layout()
    plt.show()


file1 = "output/t5-large-qqppos-1e-4-6e-gcn-decoder_attention-256-amr/epoch_1/pairwise_metrics.csv"
file2 = "output/t5-large-qqppos-1e-4-6e-linearized-amr/epoch_5/pairwise_metrics.csv"
label1 = "SAPG"
label2 = "Linearized AMR"

# Read both CSV files
df1 = pd.read_csv(file1, sep="\t")
df2 = pd.read_csv(file2, sep="\t")

# %%
# Calculate the mean metric for each source length bin for each file
mean_metric_1 = calculate_mean_metric(df1, "sbert")
mean_metric_2 = calculate_mean_metric(df2, "sbert")

# %%
plot_mean_metric(mean_metric_1, mean_metric_2, label1, label2, y_label="mean SBERT")

# %%
