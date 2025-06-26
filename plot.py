import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def plot_cluster_histogram(df_cl):
    valid = df_cl["cluster_num"].dropna().astype(int)
    counts = valid.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10,6))
    bins = np.arange(valid.min(), valid.max()+2) - 0.5
    ax.hist(valid, bins=bins, edgecolor='black', alpha=0.6)
    ax.set_xlabel("Cluster Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Cluster Frequency with Kernel Density Estimate")
    ax.set_xticks(np.arange(valid.min(), valid.max()+1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for x, freq in counts.items():
        ax.text(x, freq + 0.5, str(freq), ha='center', va='bottom')

    ax2 = ax.twinx()
    kde = gaussian_kde(valid)
    x_vals = np.linspace(valid.min(), valid.max(), 200)
    bin_width = 1
    kde_vals = kde(x_vals) * len(valid) * bin_width
    ax2.plot(x_vals, kde_vals, color='red', linewidth=2, label='KDE')
    ax2.set_ylabel("Density (scaled)")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_dataframe_info(df, title="Data Summary"):
    print(f"--- {title} ---")
    print(df.info())
    print(df.describe(include='all'))
    print(df.head())

# You can add more plotting functions here for other datasets
