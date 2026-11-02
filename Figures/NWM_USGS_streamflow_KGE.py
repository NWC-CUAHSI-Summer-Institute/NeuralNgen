import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------
# 0. Load and match gages
# ---------------------------------------------------------------
DATA_DIR = "/bighome/hdagne1/NeuralNGEN/data/streamflow"

NWM3_retrospective_1979_2023 = pd.read_parquet(f"{DATA_DIR}/NWMv3_retro_1979_2023.parquet")
USGS_streamflow_1979_2023 = pd.read_parquet(f"{DATA_DIR}/USGS_streamflow_1979_2023.parquet")

def pad_gage(cols):
    """Canonical USGS ID: digits only, no spurious leading zeros, padded to >= 8 digits."""
    s = (pd.Index(cols).astype(str).str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .str.replace(r"\D", "", regex=True))
    return s.str.lstrip("0").str.zfill(8)

def region_counts(ids, name):
    """Gage counts by first two digits (USGS region code)."""
    c = pd.Series(pd.Index(ids).str[:2]).value_counts().sort_index()
    print(f"{name:<22}", dict(c))

# raw IDs before normalization (look for 9-digit '01x...' IDs)
for name, df in [("NWM raw", NWM3_retrospective_1979_2023),
                 ("USGS raw", USGS_streamflow_1979_2023)]:
    lens = pd.Series(df.columns.astype(str).str.len()).value_counts().to_dict()
    print(f"{name} ID lengths: {lens}")

USGS_streamflow_1979_2023.columns = pad_gage(USGS_streamflow_1979_2023.columns)
NWM3_retrospective_1979_2023.columns = pad_gage(NWM3_retrospective_1979_2023.columns)

for name in ["NWM", "USGS"]:
    df = NWM3_retrospective_1979_2023 if name == "NWM" else USGS_streamflow_1979_2023
    n_dup = int(df.columns.duplicated().sum())
    if n_dup:
        print(f"{name}: {n_dup} duplicate IDs after normalization -> keeping first")
NWM3_retrospective_1979_2023 = NWM3_retrospective_1979_2023.loc[
    :, ~NWM3_retrospective_1979_2023.columns.duplicated()]
USGS_streamflow_1979_2023 = USGS_streamflow_1979_2023.loc[
    :, ~USGS_streamflow_1979_2023.columns.duplicated()]

common = USGS_streamflow_1979_2023.columns.intersection(
    NWM3_retrospective_1979_2023.columns, sort=False)
print(f"NWM gages: {NWM3_retrospective_1979_2023.shape[1]}")
print(f"USGS gages: {USGS_streamflow_1979_2023.shape[1]}")
print(f"Common gages: {len(common)} | USGS not in NWM: "
      f"{USGS_streamflow_1979_2023.shape[1] - len(common)}")

region_counts(NWM3_retrospective_1979_2023.columns, "NWM")
region_counts(USGS_streamflow_1979_2023.columns, "USGS")
region_counts(common, "common")

NWM_common = NWM3_retrospective_1979_2023[common]
USGS_common = USGS_streamflow_1979_2023[common]

# align time axis
NWM_common, USGS_common = NWM_common.align(USGS_common, join="inner", axis=0)
print(NWM_common.shape, USGS_common.shape,
      NWM_common.index.min(), NWM_common.index.max())

# units sanity check (NWM = m3/s; USGS often cfs -> ratio ~0.028 if not converted)
ratio = (NWM_common.median() / USGS_common.median()).replace([np.inf, -np.inf], np.nan)
print(f"median NWM/USGS ratio across gages: {ratio.median():.3f}")

# ---------------------------------------------------------------
# 1. Metrics
# ---------------------------------------------------------------
def _pair(sim, obs):
    mask = sim.notna() & obs.notna()
    return sim[mask], obs[mask]

def kge(sim, obs):
    sim, obs = _pair(sim, obs)
    if len(sim) < 2 or obs.std() == 0 or sim.std() == 0 or obs.mean() == 0:
        return np.nan
    r = np.corrcoef(sim, obs)[0, 1]
    if np.isnan(r):
        return np.nan
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def nse(sim, obs):
    sim, obs = _pair(sim, obs)
    denom = np.sum((obs - obs.mean())**2)
    if len(sim) < 2 or denom == 0:
        return np.nan
    return 1 - np.sum((sim - obs)**2) / denom

def pbias(sim, obs):
    sim, obs = _pair(sim, obs)
    if len(sim) < 2 or obs.sum() == 0:
        return np.nan
    return 100 * (sim.sum() - obs.sum()) / obs.sum()

def rmse(sim, obs):
    sim, obs = _pair(sim, obs)
    if len(sim) < 2:
        return np.nan
    return np.sqrt(np.mean((sim - obs)**2))

def correlation(sim, obs):
    sim, obs = _pair(sim, obs)
    if len(sim) < 2 or sim.std() == 0 or obs.std() == 0:
        return np.nan
    return np.corrcoef(sim, obs)[0, 1]

results = []
for gage in common:
    sim, obs = NWM_common[gage], USGS_common[gage]
    results.append({
        "gage_id": gage,
        "KGE": kge(sim, obs),
        "NSE": nse(sim, obs),
        "PBIAS": pbias(sim, obs),
        "RMSE": rmse(sim, obs),
        "r": correlation(sim, obs),
        "n_obs": int((sim.notna() & obs.notna()).sum()),
    })

metrics_df = pd.DataFrame(results).set_index("gage_id")
print(metrics_df.shape)
print(f"Gages with valid KGE: {metrics_df['KGE'].notna().sum()}")
print(f"Gages with NaN KGE (zero variance/no overlap): {metrics_df['KGE'].isna().sum()}")
region_counts(metrics_df.index[metrics_df["KGE"].notna()], "valid KGE")
print(metrics_df.describe())
print(metrics_df.sort_values("KGE", ascending=False).head(10))

# ---------------------------------------------------------------
# 2. Gage coordinates from NWM metadata (cached after first run)
# ---------------------------------------------------------------
COORD_CACHE = f"{DATA_DIR}/nwm3_gage_coords.csv"

if os.path.exists(COORD_CACHE):
    coord_lookup = pd.read_csv(COORD_CACHE, dtype={"gage_id": str})
else:
    ds = xr.open_zarr("s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
                      storage_options={"anon": True})
    coord_lookup = pd.DataFrame({
        "gage_id": np.char.strip(ds["gage_id"].values.astype(str)),
        "lat": ds["latitude"].values,
        "lon": ds["longitude"].values,
    })
    coord_lookup = coord_lookup[coord_lookup["gage_id"] != ""]
    coord_lookup.to_csv(COORD_CACHE, index=False)

coord_lookup = coord_lookup.dropna(subset=["gage_id"])
coord_lookup["gage_id"] = pad_gage(coord_lookup["gage_id"])
coord_lookup = coord_lookup[coord_lookup["gage_id"].str.len() > 0]
coord_lookup = coord_lookup.drop_duplicates("gage_id")

metrics_geo = metrics_df.reset_index().merge(coord_lookup, on="gage_id", how="left")
no_xy = metrics_geo["lat"].isna()
print(f"Gages without coordinates: {no_xy.sum()}")
region_counts(metrics_geo.loc[no_xy, "gage_id"], "missing coords")
region_counts(metrics_geo.dropna(subset=["lat", "lon", "KGE"])["gage_id"], "plotted")

# ---------------------------------------------------------------
# 3. Shared KGE bins + colors (single source for map AND bar chart)
# ---------------------------------------------------------------
VMIN, VMAX, NBINS = -1, 1, 5
CMAP = plt.get_cmap("gist_rainbow")
levels = np.linspace(VMIN, VMAX, NBINS + 1)
norm = BoundaryNorm(levels, CMAP.N, extend="min")

# bins: (-inf, VMIN) + the NBINS colorbar bins; left-closed like BoundaryNorm,
# top edge open to +inf so KGE = 1 falls in the last bin
edges = np.concatenate(([-np.inf], levels[:-1], [np.inf]))
labels = [f"< {VMIN:.1f}"] + [f"{levels[i]:.1f}–{levels[i+1]:.1f}" for i in range(NBINS)]

# bar colors sampled from the same cmap/norm as the map points
reps = np.concatenate(([VMIN - 1.0], 0.5 * (levels[:-1] + levels[1:])))
colors = [CMAP(norm(v)) for v in reps]

metrics_geo["KGE_bin"] = pd.cut(metrics_geo["KGE"], bins=edges, labels=labels, right=False)
n_nodata = int(metrics_geo["KGE"].isna().sum())
n_total = len(metrics_geo)
bin_counts = metrics_geo["KGE_bin"].value_counts().reindex(labels).fillna(0).astype(int)

# ---------------------------------------------------------------
# 4. Map + bar chart
# ---------------------------------------------------------------
fig = plt.figure(figsize=(12, 9))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.2], hspace=0.08)
ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
ax_bar = fig.add_subplot(gs[1])

ax_map.add_feature(cfeature.LAND, facecolor="whitesmoke")
ax_map.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax_map.add_feature(cfeature.STATES, edgecolor="gray", linewidth=0.5)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax_map.add_feature(cfeature.BORDERS, linewidth=0.6)

plot_pts = metrics_geo.dropna(subset=["lat", "lon", "KGE"])
ax_map.scatter(plot_pts["lon"], plot_pts["lat"], c=plot_pts["KGE"],
               cmap=CMAP, norm=norm, s=8, alpha=0.9,
               transform=ccrs.PlateCarree(), zorder=5)
ax_map.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax_map, label="KGE",
                  shrink=0.73, pad=0.02, extend="min", ticks=levels)
cb.set_ticklabels([f"{x:.1f}" for x in levels])
ax_map.set_title(f"NWM v3.0 vs. USGS: KGE by gage location across CONUS (n={n_total} gages)",
                 loc="center")

ax_bar.bar(labels, bin_counts.values, color=colors, edgecolor="black")
ymax = max(bin_counts.max(), 1)
for i, v in enumerate(bin_counts.values):
    ax_bar.text(i, v + ymax * 0.02, f"{v}\n({100 * v / n_total:.0f}%)",
                ha="center", va="bottom", fontsize=9)

ax_bar.set_ylim(0, ymax * 1.25)
ax_bar.set_xlabel("KGE")
ax_bar.set_ylabel("Number of gages")
ax_bar.set_title(f"KGE distribution (n={n_total - n_nodata} valid, {n_nodata} no data)",
                 fontsize=13, loc="left")
ax_bar.tick_params(axis="x", rotation=45, labelsize=10)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

plt.tight_layout()
# plt.savefig(f"{DATA_DIR}/kge_spatial_map_with_summary.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 5. Summary table
# ---------------------------------------------------------------
summary = bin_counts.to_frame("n_gages")
summary["pct"] = 100 * summary["n_gages"] / n_total
print(summary)
print(f"\nTotal gages: {n_total}")
print(f"No data (NaN KGE): {n_nodata}")