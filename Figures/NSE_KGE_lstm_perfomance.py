import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    from mintpy.objects.colors import ColormapExt
    cmap = ColormapExt("RdYlGn").colormap
except ImportError:
    print("mintpy not available, falling back to standard 'bwr' colormap")
    cmap = plt.get_cmap("RdYlGn")


def make_boundary_norm(values, n_bins=6):
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if vmin == vmax:
        pad = 0.5 if vmin == 0 else abs(vmin) * 0.05
        vmin, vmax = vmin - pad, vmax + pad
    edges = np.linspace(vmin, vmax, n_bins + 1)
    return mcolors.BoundaryNorm(edges, cmap.N)


fold_files = {
    0: "/bighome/hdagne1/NeuralNGEN/runs/cv_fold0_fdc/holdout_metrics.csv",
    1: "/bighome/hdagne1/NeuralNGEN/runs/cv_fold1_fdc/holdout_metrics.csv",
    2: "/bighome/hdagne1/NeuralNGEN/runs/cv_fold2_fdc/holdout_metrics.csv",
}
combined = pd.concat([pd.read_csv(p, dtype={"gauge_id": str}) for p in fold_files.values()], ignore_index=True)

topo = pd.read_csv("/bighome/hdagne1/NeuralNGEN/Data/CAMELS_US/camels_attributes_v2.0/camels_topo.txt",
                    sep=";", dtype={"gauge_id": str})
combined = combined.merge(topo[["gauge_id", "gauge_lat", "gauge_lon"]], on="gauge_id", how="left")

fig = plt.figure(figsize=(6, 4), dpi=200)

for i, (metric, clip) in enumerate([("NSE", (-1, 1)), ("KGE", (-1, 1))], start=1):
    ax = fig.add_subplot(2, 1, i, projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")

    ax.add_feature(cfeature.LAND, facecolor="#f3efe1")
    ax.add_feature(cfeature.OCEAN, facecolor="#cfe3f0")
    ax.add_feature(cfeature.LAKES, facecolor="#cfe3f0", edgecolor="#8fa8b8", linewidth=0.4)
    ax.add_feature(cfeature.STATES, edgecolor="#888888", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#555555", linewidth=0.7)
    ax.add_feature(cfeature.RIVERS, edgecolor="#7fb3d5", linewidth=0.4, alpha=0.5)

    vals = combined[metric].clip(*clip)
    norm = make_boundary_norm(vals.dropna().values, n_bins=6)

    sc = ax.scatter(combined["gauge_lon"], combined["gauge_lat"], c=vals, cmap=cmap, norm=norm,
                     s=10, edgecolor="k", linewidth=0.3,
                     transform=ccrs.PlateCarree(), zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=metric, shrink=0.96, pad=0.02, extend='both')
    cbar.set_label(metric, fontsize=14)
    ax.set_title(f"{metric} (n={combined[metric].notna().sum()})", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("all_basins_nse_kge_map.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"NSE: median={combined['NSE'].median():.3f}, range=[{combined['NSE'].min():.2f}, {combined['NSE'].max():.2f}]")
print(f"KGE: median={combined['KGE'].median():.3f}, range=[{combined['KGE'].min():.2f}, {combined['KGE'].max():.2f}]")