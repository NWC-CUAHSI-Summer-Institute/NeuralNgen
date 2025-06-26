#!/usr/bin/env python3
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from pathlib import Path
import sys
# --- CONFIG ---
OWNER       = "NWC-CUAHSI-Summer-Institute"
REPO        = "CAMELS_data_sample"
BRANCH      = "main"
DIRS = {
    "attributes": "camels_attributes_v2.0",
    "hourly":     "hourly/aorc_hourly"
}
CATS_PATH   = "categoriesinformation"
LOCAL_CATS  = ["categoriesinformation", "categoriesinformation.txt"]
# --- HELPERS ---
def list_github_dir(owner, repo, path, branch="main"):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    resp = requests.get(url, params={"ref": branch})
    resp.raise_for_status()
    return resp.json()
def load_table_from_url(url):
    txt = requests.get(url).text
    buf = StringIO(txt)
    try:
        return pd.read_csv(buf, engine='python', sep=None, on_bad_lines='warn')
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf, engine='python', sep=r'\s+', on_bad_lines='warn')
#Building the Master Dictionary
def build_data_dict():
    data = {}
    # 1) Load attribute & hourly tables
    for section, path in DIRS.items():
        try:
            entries = list_github_dir(OWNER, REPO, path, BRANCH)
        except Exception:
            data[section] = {}
            continue
        data[section] = {}
        for entry in entries:
            name = entry["name"]
            dl   = entry.get("download_url")
            if dl and name.lower().endswith((".csv", ".txt", ".dat")):
                df = load_table_from_url(dl)
                data[section][name] = df
    # 2) Load categoriesinformation (clusters)
    txt = None
    for fname in LOCAL_CATS:
        p = Path(fname)
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            break
    if txt is None:
        print("No local 'categoriesinformation' found!", file=sys.stderr)
        sys.exit(1)
    clusters = [
        line.strip()
        for line in txt.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    data["clusters"] = clusters
    return data
def main():
    master_dict = build_data_dict()
    # 3) Inspect DataFrames
    for section, content in master_dict.items():
        if section == "clusters":
            continue
        print(f"\n=== Section: {section} ({len(content)} files) ===")
        for fname, df in content.items():
            print(f"\n--- File: {fname} ---")
            print(df.info())
            print("\nFirst 5 rows:")
            print(df.head().to_string(index=False))
            print("\nSummary stats:")
            print(df.describe(include='all').to_string())
    # 4) Parse gauge_id → cluster pairs
    raw = master_dict["clusters"]
    pairs = []
    for line in raw:
        if ";" in line:
            parts = [p.strip() for p in line.split(";")]
        else:
            parts = line.split()
        pairs.append((parts[0], parts[-1]))
    df_cl = pd.DataFrame(pairs, columns=["gauge_id", "cluster"])
    df_cl["cluster_num"] = pd.to_numeric(df_cl["cluster"], errors="coerce")
    valid = df_cl["cluster_num"].dropna().astype(int)
    # 5) Compute frequency counts
    counts = valid.value_counts().sort_index()
    # 6) Plot frequency histogram
    fig, ax = plt.subplots(figsize=(10,6))
    bins = np.arange(valid.min(), valid.max()+2) - 0.5
    ax.hist(valid, bins=bins, edgecolor='black', alpha=0.6)
    ax.set_xlabel("Cluster Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Cluster Frequency with Kernel Density Estimate")
    ax.set_xticks(np.arange(valid.min(), valid.max()+1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Annotate counts
    for x, freq in counts.items():
        ax.text(x, freq + 0.5, str(freq), ha='center', va='bottom')
    # 7) Overlay KDE on secondary y-axis
    ax2 = ax.twinx()
    # compute KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(valid)
    x_vals = np.linspace(valid.min(), valid.max(), 200)
    # scale KDE to match histogram area
    bin_width = 1
    kde_vals = kde(x_vals) * len(valid) * bin_width
    ax2.plot(x_vals, kde_vals, color='red', linewidth=2, label='KDE')
    ax2.set_ylabel("Density (scaled)")
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()