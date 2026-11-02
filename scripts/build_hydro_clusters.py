#!/usr/bin/env python3
"""
recluster_basins.py
====================
Re-cluster all 440 ready basins from scratch using GAGES-II static
climate/physiographic attributes, replacing hydro_clusters.csv entirely.

The original clustering (10 clusters, 280 gauges) used an unknown method/
feature set - it was never recovered. Rather than guess at extending it
inconsistently for the ~160 newly-added gauges, this re-fits k-means on
ALL 440 gauges together using GAGES-II basin-characteristic features,
producing one consistent, from-scratch clustering.

Features used (all from gages_ii.parquet, standardized before k-means):
  - PPTAVG_BASIN, T_AVG_BASIN, PET, ARIDITY, SNOW_PCT_PRECIP  (climate)
  - RUNAVE7100, BFI_AVE                                        (flow regime proxies)
  - PERMAVE, CLAYAVE, SANDAVE, SILTAVE, AWCAVE, WTDEPAVE        (soil)
  - ELEV_MEAN_M_BASIN, SLOPE_PCT                                (terrain)
  - DRAIN_SQKM (log-transformed - heavily right-skewed otherwise)
  - dec_lat_va, dec_long_va                                     (rough geography)

k=10 to match the original cluster count (and your configs' spatial_length
assumption), not because 10 is provably optimal - change --n-clusters if
you want to test a different k.

Usage
-----
    python recluster_basins.py \\
        --basin-list /bighome/hdagne1/Docker/NeuralNGEN/basin_lists/ready_basins_filtered_updated.txt \\
        --gages-ii /bighome/hdagne1/conus-lowflow-hydro-drought-thresholds/data/gage-II/gages_ii.parquet \\
        --out-csv /bighome/hdagne1/Docker/NeuralNGEN/data/hydro_clusters.csv \\
        --n-clusters 10
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "PPTAVG_BASIN", "T_AVG_BASIN", "PET", "ARIDITY", "SNOW_PCT_PRECIP",
    "RUNAVE7100", "BFI_AVE",
    "PERMAVE", "CLAYAVE", "SANDAVE", "SILTAVE", "AWCAVE", "WTDEPAVE",
    "ELEV_MEAN_M_BASIN", "SLOPE_PCT",
    "DRAIN_SQKM",
    "dec_lat_va", "dec_long_va",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basin-list", type=Path, required=True)
    ap.add_argument("--gages-ii", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gauges = [l.strip().zfill(8) for l in args.basin_list.read_text().splitlines() if l.strip()]
    print(f"Basins to cluster: {len(gauges)}")

    gages_ii = pd.read_parquet(args.gages_ii)
    gages_ii["site_no"] = gages_ii["site_no"].astype(str).str.zfill(8)
    gages_ii = gages_ii.set_index("site_no")

    missing = [g for g in gauges if g not in gages_ii.index]
    if missing:
        print(f"WARNING: {len(missing)} gauges not found in GAGES-II, will be excluded: {missing[:20]}")
    present = [g for g in gauges if g in gages_ii.index]

    feat_df = gages_ii.loc[present, FEATURES].copy()

    # Some GAGES-II columns come through as strings (Arrow-backed dtypes,
    # or values with stray whitespace) rather than numeric - coerce
    # everything explicitly before any arithmetic. Anything that fails to
    # parse becomes NaN and gets median-imputed below, same as a genuine
    # missing value.
    for col in FEATURES:
        feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")

    nan_counts = feat_df.isna().sum()
    if nan_counts.sum() > 0:
        print("NaN counts per feature (median-imputed):")
        print(nan_counts[nan_counts > 0])
        feat_df = feat_df.fillna(feat_df.median())

    # Log-transform drainage area - heavily right-skewed, would otherwise
    # dominate the distance metric for a handful of very large basins.
    feat_df["DRAIN_SQKM"] = np.log1p(feat_df["DRAIN_SQKM"])

    X = StandardScaler().fit_transform(feat_df.values)

    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    labels = km.fit_predict(X)

    out_df = pd.DataFrame({"gauge_id": present, "Clusters": labels})
    out_df.to_csv(args.out_csv, index=False)

    print(f"\nCluster sizes:")
    print(out_df["Clusters"].value_counts().sort_index())
    print(f"\nSaved {len(out_df)} gauge->cluster assignments to {args.out_csv}")

    if missing:
        print(f"\n{len(missing)} gauges could not be clustered (not in GAGES-II) and are "
              f"NOT in the output file - they'll be silently skipped by MultiGaugeBatcher.")


if __name__ == "__main__":
    main()





# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# SIGNATURE_COLS = [
#     "q_mean", "runoff_ratio", "slope_fdc", "baseflow_index", "stream_elas",
#     "q5", "q95", "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur",
#     "zero_q_freq", "hfd_mean",
# ]

# N_CLUSTERS = 10  # matches paper Figure 1

# basin_list_path = "/bighome/hdagne1/Docker/NeuralNGEN/basin_lists/ready_basins_filtered.txt"
# hydro_path = "/bighome/hdagne1/conus-national-low-flow-thresholds-lstm/data/CAMELS_US/camels_attributes_v2.0/camels_hydro.txt"
# out_path = "/bighome/hdagne1/Docker/NeuralNGEN/data/hydro_clusters.csv"

# basins = [b.strip() for b in open(basin_list_path) if b.strip()]
# print(f"Clustering {len(basins)} extracted basins")

# hydro = pd.read_csv(hydro_path, sep=";", dtype={"gauge_id": str})
# hydro = hydro.set_index("gauge_id")

# missing = [b for b in basins if b not in hydro.index]
# if missing:
#     print(f"WARNING: {len(missing)} basins missing from camels_hydro.txt, dropping: {missing}")
# basins = [b for b in basins if b in hydro.index]

# sig_df = hydro.loc[basins, SIGNATURE_COLS].astype(float)

# n_nan = sig_df.isna().sum().sum()
# if n_nan > 0:
#     print(f"WARNING: {n_nan} NaN signature values, filling with column mean")
#     sig_df = sig_df.fillna(sig_df.mean())

# scaler = StandardScaler()
# sig_scaled = scaler.fit_transform(sig_df.values)

# kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
# cluster_labels = kmeans.fit_predict(sig_scaled)

# result = pd.DataFrame({"gauge_id": basins, "Clusters": cluster_labels})
# result.to_csv(out_path, index=False)

# print(f"\nSaved cluster assignments to {out_path}")
# print("\nCluster distribution:")
# print(result["Clusters"].value_counts().sort_index())