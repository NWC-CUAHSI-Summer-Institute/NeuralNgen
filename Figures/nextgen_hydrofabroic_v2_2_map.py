
"""
Map the full CONUS NextGen hydrofabric (v2.2): divides, flowpaths, gauged hydrolocations.

Jupyter:
    run_map(gpkg="/path/conus_nextgen.gpkg", min_order=3)

Command line:
    python map_conus_hydrofabric.py --gpkg /path/conus_nextgen.gpkg --min-order 3

Requirements:
    pip install geopandas pyogrio pyarrow shapely matplotlib requests
"""

import argparse
import os
import sys
import tarfile
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyogrio
import requests
from matplotlib.collections import LineCollection, PolyCollection
from shapely import get_coordinates, get_parts

# CIROH community hydrofabric (v2.2 CONUS, used by NGIAB data preprocess).
# The former Lynker-Spatial public URL now returns HTTP 403.
HF_URL = ("https://communityhydrofabric.s3.us-east-1.amazonaws.com/"
          "hydrofabrics/community/conus_nextgen.tar.gz")
NGIAB_GPKG = os.path.expanduser("~/.ngiab/hydrofabric/v2.2/conus_nextgen.gpkg")
CRS = "EPSG:5070"  # CONUS Albers; native CRS of the v2.2 hydrofabric


def _stream(url, out):
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code in (401, 403):
            raise RuntimeError(f"Access denied (HTTP {r.status_code}) for {url}")
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        tmp = out + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e9:6.2f} / {total/1e9:.2f} GB", end="")
        print()
        os.replace(tmp, out)


def get_gpkg(gpkg):
    """Return a path to conus_nextgen.gpkg: given path, NGIAB cache, or download."""
    if os.path.exists(gpkg):
        return gpkg
    if os.path.exists(NGIAB_GPKG):
        print(f"[found] NGIAB copy: {NGIAB_GPKG}")
        return NGIAB_GPKG

    out_dir = os.path.dirname(os.path.abspath(gpkg))
    os.makedirs(out_dir, exist_ok=True)
    tgz = os.path.join(out_dir, "conus_nextgen.tar.gz")
    if not os.path.exists(tgz):
        _stream(HF_URL, tgz)

    print(f"[extract] {tgz}")
    with tarfile.open(tgz, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".gpkg")]
        if not members:
            raise RuntimeError("No .gpkg found in archive")
        m = members[0]
        m.name = os.path.basename(m.name)  # flatten; avoid path traversal
        tf.extract(m, out_dir)
        extracted = os.path.join(out_dir, m.name)

    if os.path.abspath(extracted) != os.path.abspath(gpkg):
        os.replace(extracted, gpkg)
    print(f"[ready] {gpkg}")
    return gpkg


def pick_layer(layers, *candidates):
    names = {n.lower(): n for n in layers}
    for c in candidates:
        if c.lower() in names:
            return names[c.lower()]
    return None


def read(gpkg, layer, columns=None):
    t = time.time()
    df = gpd.read_file(gpkg, layer=layer, engine="pyogrio",
                       use_arrow=True, columns=columns)
    if df.crs is not None and df.crs.to_string() != CRS:
        df = df.to_crs(CRS)
    print(f"[read] {layer}: {len(df):,} features ({time.time()-t:.0f} s)")
    return df


def polygon_rings(geoms):
    """Exterior rings of all polygons as a list of (N,2) arrays."""
    rings = []
    for poly in get_parts(geoms):
        if poly is None or poly.is_empty:
            continue
        rings.append(np.asarray(poly.exterior.coords)[:, :2])
    return rings


def line_segments(geoms):
    """All linestring parts as a list of (N,2) arrays."""
    return [get_coordinates(g) for g in get_parts(geoms)
            if g is not None and not g.is_empty]


def run_map(gpkg="conus_nextgen.gpkg", out="conus_hydrofabric_v2.2.png",
            dpi=400, simplify=100.0, min_order=0, show=True):
    """
    gpkg      : path to conus_nextgen.gpkg (uses ~/.ngiab copy or downloads if missing)
    out       : output PNG path
    dpi       : output resolution
    simplify  : simplification tolerance in metres (0 = none)
    min_order : plot only flowpaths with stream order >= this (0 = all)
    show      : display the figure inline (Jupyter)
    """
    gpkg = get_gpkg(gpkg)

    layers = [l[0] for l in pyogrio.list_layers(gpkg)]
    print("[layers]", ", ".join(layers))
    lyr_div = pick_layer(layers, "divides")
    lyr_fp = pick_layer(layers, "flowpaths", "flowlines")
    lyr_hl = pick_layer(layers, "hydrolocations")
    if lyr_div is None or lyr_fp is None:
        raise RuntimeError("divides/flowpaths layers not found")

    # ---------- divides ----------
    div_cols = [c for c in pyogrio.read_info(gpkg, layer=lyr_div)["fields"]
                if c.lower() in ("divide_id", "vpuid")]
    div = read(gpkg, lyr_div, columns=div_cols)
    if simplify > 0:
        div["geometry"] = div.geometry.simplify(simplify, preserve_topology=True)

    # ---------- flowpaths ----------
    fp_fields = pyogrio.read_info(gpkg, layer=lyr_fp)["fields"]
    order_col = next((c for c in fp_fields
                      if c.lower() in ("order", "streamorder", "stream_order")), None)
    fp = read(gpkg, lyr_fp, columns=[order_col] if order_col else None)
    if min_order > 0 and order_col:
        fp = fp[fp[order_col] >= min_order]
        print(f"[filter] flowpaths with {order_col} >= {min_order}: {len(fp):,}")
    if simplify > 0:
        fp["geometry"] = fp.geometry.simplify(simplify, preserve_topology=False)

    # ---------- gauges (USGS hydrolocations) ----------
    gauges = None
    if lyr_hl is not None:
        hl = read(gpkg, lyr_hl)
        src_col = next((c for c in hl.columns
                        if c.lower() in ("hl_reference", "hl_source")), None)
        if src_col is not None:
            gauges = hl[hl[src_col].astype(str).str.contains("gage|usgs", case=False, na=False)]
        else:
            gauges = hl
        print(f"[gauges] {len(gauges):,}")

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(20, 12.5))
    ax.set_facecolor("white")

    # Divides coloured by VPU
    vpu_col = next((c for c in div.columns if c.lower() == "vpuid"), None)
    if vpu_col:
        vpus = sorted(div[vpu_col].astype(str).unique())
        cmap = plt.get_cmap("tab20", len(vpus))
        for i, v in enumerate(vpus):
            rings = polygon_rings(div.geometry[div[vpu_col].astype(str) == v].values)
            ax.add_collection(PolyCollection(rings, facecolor=cmap(i), alpha=0.25,
                                             edgecolor="0.55", linewidth=0.03))
    else:
        ax.add_collection(PolyCollection(polygon_rings(div.geometry.values),
                                         facecolor="0.92", edgecolor="0.55", linewidth=0.03))

    # Flowpaths: width scaled by stream order when available
    if order_col:
        orders = fp[order_col].fillna(1).astype(int)
        for o in sorted(orders.unique()):
            segs = line_segments(fp.geometry[orders == o].values)
            ax.add_collection(LineCollection(segs, colors="#1f5fa8",
                                             linewidths=0.05 + 0.12 * (o - 1), alpha=0.9))
    else:
        ax.add_collection(LineCollection(line_segments(fp.geometry.values),
                                         colors="#1f5fa8", linewidths=0.08))

    if gauges is not None and len(gauges):
        ax.scatter(gauges.geometry.x, gauges.geometry.y, s=0.6, c="#c0392b",
                   linewidths=0, label=f"USGS gauges (n={len(gauges):,})", zorder=5)
        ax.legend(loc="lower left", markerscale=12, frameon=False, fontsize=11)

    xmin, ymin, xmax, ymax = div.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"NextGen Hydrofabric v2.2 — CONUS  |  {len(div):,} divides, "
                 f"{len(fp):,} flowpaths", fontsize=15)

    t = time.time()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {out} ({time.time()-t:.0f} s)")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", default="conus_nextgen.gpkg")
    ap.add_argument("--out", default="conus_hydrofabric_v2.2.png")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--simplify", type=float, default=100.0)
    ap.add_argument("--min-order", type=int, default=0)
    args, _ = ap.parse_known_args(argv)  # ignores Jupyter's --f=kernel.json
    run_map(gpkg=args.gpkg, out=args.out, dpi=args.dpi,
            simplify=args.simplify, min_order=args.min_order,
            show="ipykernel" in sys.modules)


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()  # command line only; in Jupyter call run_map(...) yourself


run_map(gpkg="/path/with/space/conus_nextgen.gpkg", min_order=3)



















# """
# CONUS NextGen hydrofabric v2.2 map: cartopy basemap, divides by VPU, rivers by stream
# order, USGS gauges, and a zoom inset showing individual divides.

# Jupyter:
#     run_map()                                   # uses ~/.ngiab copy if present
#     run_map(gpkg="/path/conus_nextgen.gpkg", min_order=3, inset=(-71.6, -70.4, 44.4, 45.3))

# Requirements:
#     pip install geopandas pyogrio pyarrow shapely matplotlib cartopy pyproj requests
# """

# import argparse
# import os
# import sys
# import tarfile
# import time

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import numpy as np
# import pyogrio
# import requests
# from matplotlib.collections import LineCollection, PolyCollection
# from matplotlib.lines import Line2D
# from matplotlib.patches import Rectangle
# from pyproj import Transformer
# from shapely import get_coordinates, get_parts

# # ---------------------------------------------------------------------------
# # Data source
# # ---------------------------------------------------------------------------
# HF_URL = ("https://communityhydrofabric.s3.us-east-1.amazonaws.com/"
#           "hydrofabrics/community/conus_nextgen.tar.gz")
# NGIAB_GPKG = os.path.expanduser("~/.ngiab/hydrofabric/v2.2/conus_nextgen.gpkg")
# CRS = "EPSG:5070"

# # EPSG:5070 (NAD83 / CONUS Albers) expressed as a cartopy projection, so the
# # hydrofabric is drawn in its native coordinates with no reprojection.
# ALBERS = ccrs.AlbersEqualArea(central_longitude=-96.0, central_latitude=23.0,
#                               false_easting=0.0, false_northing=0.0,
#                               standard_parallels=(29.5, 45.5),
#                               globe=ccrs.Globe(ellipse="GRS80"))

# # ---------------------------------------------------------------------------
# # Style (basemap matches the KGE map: whitesmoke land, lightblue ocean, grey states)
# # ---------------------------------------------------------------------------
# LAND = "whitesmoke"
# OCEAN = "lightblue"
# STATE_EDGE = "gray"
# VPU_PALETTE = ["#e8d5b7", "#cfe3c3", "#f3d1c8", "#d9d2e9", "#fbe7a1", "#c8e0e8",
#                "#e6cfe0", "#d6e6b8", "#f6dcb2", "#cbd8ea", "#efd0d0", "#d8e8d0",
#                "#ead9c4", "#d0dcd8", "#f0e0c0", "#dcd0e8", "#cfe6dd", "#f2d6bd",
#                "#d4dce8", "#e9e2b8", "#e0cfc8"]
# RIVER_CMAP = plt.get_cmap("Blues")
# GAUGE_COLOR = "red"


# # ---------------------------------------------------------------------------
# # I/O
# # ---------------------------------------------------------------------------
# def _stream(url, out):
#     print(f"[download] {url}")
#     with requests.get(url, stream=True, timeout=60) as r:
#         if r.status_code in (401, 403):
#             raise RuntimeError(f"Access denied (HTTP {r.status_code}) for {url}")
#         r.raise_for_status()
#         total = int(r.headers.get("content-length", 0))
#         done = 0
#         tmp = out + ".part"
#         with open(tmp, "wb") as f:
#             for chunk in r.iter_content(chunk_size=8 << 20):
#                 f.write(chunk)
#                 done += len(chunk)
#                 if total:
#                     print(f"\r  {done/1e9:6.2f} / {total/1e9:.2f} GB", end="")
#         print()
#         os.replace(tmp, out)


# def get_gpkg(gpkg):
#     """Return a path to conus_nextgen.gpkg: given path, NGIAB cache, or download."""
#     if os.path.exists(gpkg):
#         return gpkg
#     if os.path.exists(NGIAB_GPKG):
#         print(f"[found] NGIAB copy: {NGIAB_GPKG}")
#         return NGIAB_GPKG
#     out_dir = os.path.dirname(os.path.abspath(gpkg))
#     os.makedirs(out_dir, exist_ok=True)
#     tgz = os.path.join(out_dir, "conus_nextgen.tar.gz")
#     if not os.path.exists(tgz):
#         _stream(HF_URL, tgz)
#     print(f"[extract] {tgz}")
#     with tarfile.open(tgz, "r:gz") as tf:
#         members = [m for m in tf.getmembers() if m.name.endswith(".gpkg")]
#         if not members:
#             raise RuntimeError("No .gpkg found in archive")
#         m = members[0]
#         m.name = os.path.basename(m.name)
#         tf.extract(m, out_dir)
#         extracted = os.path.join(out_dir, m.name)
#     if os.path.abspath(extracted) != os.path.abspath(gpkg):
#         os.replace(extracted, gpkg)
#     print(f"[ready] {gpkg}")
#     return gpkg


# def pick_layer(layers, *candidates):
#     names = {n.lower(): n for n in layers}
#     for c in candidates:
#         if c.lower() in names:
#             return names[c.lower()]
#     return None


# def read(gpkg, layer, columns=None):
#     t = time.time()
#     df = gpd.read_file(gpkg, layer=layer, engine="pyogrio", use_arrow=True, columns=columns)
#     if df.crs is not None and df.crs.to_string() != CRS:
#         df = df.to_crs(CRS)
#     print(f"[read] {layer}: {len(df):,} features ({time.time()-t:.0f} s)")
#     return df


# # ---------------------------------------------------------------------------
# # Geometry -> matplotlib collections
# # ---------------------------------------------------------------------------
# def polygon_rings(geoms):
#     rings = []
#     for poly in get_parts(geoms):
#         if poly is None or poly.is_empty:
#             continue
#         rings.append(np.asarray(poly.exterior.coords)[:, :2])
#     return rings


# def line_segments(geoms):
#     return [get_coordinates(g) for g in get_parts(geoms) if g is not None and not g.is_empty]


# def river_style(order, o_min, o_max, inset=False):
#     """Colour (light -> dark blue) and width increasing with stream order."""
#     f = 0.0 if o_max == o_min else (order - o_min) / (o_max - o_min)
#     color = RIVER_CMAP(0.45 + 0.55 * f)
#     width = (0.25 + 1.6 * f ** 1.4) if inset else (0.08 + 1.1 * f ** 1.6)
#     return color, width


# def _load_features(detail):
#     """Natural Earth features; geometries are loaded now so download errors surface here."""
#     feats = {
#         "ocean": cfeature.OCEAN.with_scale(detail),
#         "land": cfeature.LAND.with_scale(detail),
#         "lakes": cfeature.LAKES.with_scale(detail),
#         "borders": cfeature.BORDERS.with_scale(detail),
#         "coast": cfeature.COASTLINE.with_scale(detail),
#         "states": cfeature.STATES.with_scale(detail),
#     }
#     for f in feats.values():
#         next(iter(f.geometries()))
#     return feats


# def add_basemap(ax, detail="50m"):
#     feats = None
#     for d in dict.fromkeys([detail, "110m"]):
#         try:
#             feats = _load_features(d)
#             break
#         except Exception as e:
#             print(f"[warn] Natural Earth {d} unavailable: {e}")
#     if feats is None:
#         ax.set_facecolor(LAND)
#         return False
#     ax.add_feature(feats["ocean"], facecolor=OCEAN, edgecolor="none", zorder=0)
#     ax.add_feature(feats["land"], facecolor=LAND, edgecolor="none", zorder=0)
#     ax.add_feature(feats["lakes"], facecolor=OCEAN, edgecolor="none", zorder=3)
#     ax.add_feature(feats["borders"], edgecolor="0.35", linewidth=0.6, zorder=6)
#     ax.add_feature(feats["coast"], edgecolor="0.35", linewidth=0.6, zorder=6)
#     ax.add_feature(feats["states"], edgecolor=STATE_EDGE, facecolor="none",
#                    linewidth=0.5, zorder=6)
#     return True


# def draw_divides(ax, div, vpu_col, colors, edge, lw, alpha):
#     if vpu_col:
#         for v, sub in div.groupby(vpu_col):
#             ax.add_collection(PolyCollection(
#                 polygon_rings(sub.geometry.values), facecolor=colors[str(v)],
#                 edgecolor=edge, linewidth=lw, alpha=alpha, transform=ax.transData, zorder=2))
#     else:
#         ax.add_collection(PolyCollection(
#             polygon_rings(div.geometry.values), facecolor=VPU_PALETTE[0],
#             edgecolor=edge, linewidth=lw, alpha=alpha, transform=ax.transData, zorder=2))


# def draw_rivers(ax, fp, order_col, o_min, o_max, inset=False):
#     if order_col is None:
#         ax.add_collection(LineCollection(line_segments(fp.geometry.values),
#                                          colors=RIVER_CMAP(0.75), linewidths=0.2,
#                                          transform=ax.transData, zorder=4))
#         return
#     orders = fp[order_col].fillna(1).astype(int)
#     for o in sorted(orders.unique()):  # low orders first, big rivers on top
#         c, w = river_style(o, o_min, o_max, inset)
#         ax.add_collection(LineCollection(line_segments(fp.geometry[orders == o].values),
#                                          colors=[c], linewidths=w, capstyle="round",
#                                          joinstyle="round", transform=ax.transData,
#                                          zorder=4))


# # ---------------------------------------------------------------------------
# # Main map
# # ---------------------------------------------------------------------------
# def run_map(gpkg="conus_nextgen.gpkg", out="conus_hydrofabric_v2.2.png", dpi=300,
#             simplify=150.0, min_order=3, show_gauges=True,
#             inset=(-71.6, -70.4, 44.4, 45.3), basemap_detail="50m", show=True):
#     """
#     gpkg           : path to conus_nextgen.gpkg (uses ~/.ngiab copy or downloads if missing)
#     out            : output PNG path
#     dpi            : output resolution
#     simplify       : main-map simplification tolerance in metres (inset is unsimplified)
#     min_order      : main-map rivers with stream order >= this (inset shows all orders)
#     show_gauges    : plot USGS gauge hydrolocations
#     inset          : (lon_min, lon_max, lat_min, lat_max) zoom window, or None
#                      default = headwaters around USGS 01052500 (Diamond River, NH)
#     basemap_detail : Natural Earth scale "110m", "50m" or "10m"
#     show           : display inline (Jupyter)
#     """
#     gpkg = get_gpkg(gpkg)
#     layers = [l[0] for l in pyogrio.list_layers(gpkg)]
#     lyr_div = pick_layer(layers, "divides")
#     lyr_fp = pick_layer(layers, "flowpaths", "flowlines")
#     lyr_hl = pick_layer(layers, "hydrolocations")

#     # --- divides
#     div_fields = pyogrio.read_info(gpkg, layer=lyr_div)["fields"]
#     vpu_col = next((c for c in div_fields if c.lower() in ("vpuid", "vpu", "vpu_id")), None)
#     div = read(gpkg, lyr_div, columns=[vpu_col] if vpu_col else [])

#     # --- flowpaths
#     fp_fields = pyogrio.read_info(gpkg, layer=lyr_fp)["fields"]
#     order_col = next((c for c in fp_fields
#                       if c.lower() in ("order", "streamorder", "stream_order")), None)
#     fp_all = read(gpkg, lyr_fp, columns=[order_col] if order_col else [])
#     if order_col:
#         o_vals = fp_all[order_col].dropna().astype(int)
#         o_min_all, o_max = int(o_vals.min()), int(o_vals.max())
#         fp = fp_all[fp_all[order_col] >= min_order].copy()
#         print(f"[filter] main map: {order_col} >= {min_order}: {len(fp):,} flowpaths")
#     else:
#         o_min_all = o_max = 1
#         fp = fp_all.copy()

#     # --- gauges
#     gauges = None
#     if show_gauges and lyr_hl:
#         hl = read(gpkg, lyr_hl)
#         src = next((c for c in hl.columns if c.lower() in ("hl_reference", "hl_source")), None)
#         gauges = hl[hl[src].astype(str).str.contains("gage|usgs", case=False, na=False)] if src else hl
#         gauges = gauges[~gauges.geometry.duplicated()]
#         print(f"[gauges] {len(gauges):,}")

#     # --- VPU colours
#     if vpu_col:
#         vpus = sorted(div[vpu_col].astype(str).unique())
#         colors = {v: VPU_PALETTE[i % len(VPU_PALETTE)] for i, v in enumerate(vpus)}
#     else:
#         colors = {}

#     # --- inset subsets (unsimplified), taken before simplification
#     box = None
#     if inset is not None:
#         tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
#         lon0, lon1, lat0, lat1 = inset
#         xs, ys = tr.transform([lon0, lon1, lon0, lon1], [lat0, lat0, lat1, lat1])
#         box = (min(xs), max(xs), min(ys), max(ys))
#         div_in = div.cx[box[0]:box[1], box[2]:box[3]]
#         fp_in = fp_all.cx[box[0]:box[1], box[2]:box[3]]
#         g_in = gauges.cx[box[0]:box[1], box[2]:box[3]] if gauges is not None else None

#     t = time.time()
#     if simplify > 0:
#         div["geometry"] = div.geometry.simplify(simplify, preserve_topology=True)
#         fp["geometry"] = fp.geometry.simplify(simplify, preserve_topology=False)
#         print(f"[simplify] {time.time()-t:.0f} s")

#     # ------------------------------------------------------------------ figure
#     plt.rcParams.update({"font.family": "DejaVu Sans"})
#     fig = plt.figure(figsize=(18, 8), facecolor="white")
#     ax = fig.add_axes([0.01, 0.06, 0.74, 0.86], projection=ALBERS)
#     xmin, ymin, xmax, ymax = div.total_bounds
#     pad = 0.015 * (xmax - xmin)
#     ax.set_extent([xmin - pad, xmax + pad, ymin - pad, ymax + pad], crs=ALBERS)
#     add_basemap(ax, basemap_detail)

#     t = time.time()
#     draw_divides(ax, div, vpu_col, colors, edge=(1, 1, 1, 0.35), lw=0.04, alpha=0.95)
#     draw_rivers(ax, fp, order_col, min_order, o_max)
#     if gauges is not None and len(gauges):
#         ax.scatter(gauges.geometry.x, gauges.geometry.y, s=2, c=GAUGE_COLOR,
#                    linewidths=0.01, alpha=0.8, transform=ALBERS, zorder=7)
#     print(f"[draw] main map {time.time()-t:.0f} s")
#     ax.spines["geo"].set_edgecolor("0.6")

#     if box is not None:
#         ax.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
#                                fill=False, edgecolor="black", linewidth=1.3,
#                                transform=ALBERS, zorder=8))

#     # Title block
#     fig.text(0.15, 0.955, "NextGen Hydrofabric v2.2 — Contiguous United States",
#              fontsize=21, weight="bold", color="0.12")
#     fig.text(0.15, 0.925,
#              f"{len(div):,} divides (catchments)  ·  {len(fp_all):,} flowpaths  ·  "
#              f"{len(gauges) if gauges is not None else 0:,} USGS gauge hydrolocations",
#              fontsize=12.5, color="0.35")

#     # ------------------------------------------------------------------ inset
#     if box is not None:
#         axi = fig.add_axes([0.765, 0.43, 0.22, 0.46], projection=ALBERS)
#         axi.set_extent([box[0], box[1], box[2], box[3]], crs=ALBERS)
#         axi.set_facecolor(LAND)
#         draw_divides(axi, div_in, vpu_col, colors, edge="0.25", lw=0.45, alpha=1.0)
#         draw_rivers(axi, fp_in, order_col, o_min_all, o_max, inset=True)
#         if g_in is not None and len(g_in):
#             axi.scatter(g_in.geometry.x, g_in.geometry.y, s=38, c=GAUGE_COLOR,
#                         edgecolors="white", linewidths=0.9, transform=ALBERS, zorder=7)
#         axi.spines["geo"].set_edgecolor("black")
#         axi.spines["geo"].set_linewidth(1.3)
#         axi.set_title(f"Selected region exmpale: {len(div_in):,} divides, all stream orders",
#                       fontsize=11, color="0.25", loc="left", pad=6)
#         # scale bar (10 km)
#         sx = box[0] + 0.06 * (box[1] - box[0])
#         sy = box[2] + 0.05 * (box[3] - box[2])
#         axi.plot([sx, sx + 10_000], [sy, sy], color="black", lw=3, transform=ALBERS,
#                  zorder=9, solid_capstyle="butt")
#         axi.text(sx + 5_000, sy + 0.025 * (box[3] - box[2]), "10 km", ha="center",
#                  fontsize=9, transform=ALBERS, zorder=9)

#     # ------------------------------------------------------------------ legend
#     handles = []
#     if order_col:
#         for o in sorted({min_order, (min_order + o_max) // 2, o_max}):
#             c, w = river_style(o, min_order, o_max)
#             handles.append(Line2D([], [], color=c, lw=max(w * 2.2, 0.8),
#                                   label=f"Stream order {o}"))
#     else:
#         handles.append(Line2D([], [], color=RIVER_CMAP(0.75), lw=1, label="Flowpaths"))
#     handles.append(Line2D([], [], color="none", marker="s", markersize=11,
#                           markerfacecolor=VPU_PALETTE[0], markeredgecolor="0.4",
#                           label="Divides (coloured by VPU)" if vpu_col else "Divides"))
#     if gauges is not None and len(gauges):
#         handles.append(Line2D([], [], color="none", marker="o", markersize=6,
#                               markerfacecolor=GAUGE_COLOR, markeredgecolor="white",
#                               label="USGS gauge"))
#     if box is not None:
#         handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black",
#                                  linewidth=1.3, label="Zoom area"))
#     fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.765, 0.40),
#                frameon=False, fontsize=11.5, handlelength=2.6, labelspacing=0.9,
#                title="Legend", title_fontproperties={"weight": "bold", "size": 12.5},
#                alignment="left")

#     fig.text(0.765, 0.07,
#              "Hydrofabric: NOAA OWP / Lynker v2.2 (CIROH community copy)\n"
#              "Basemap: Natural Earth  ·  Projection: CONUS Albers (EPSG:5070)",
#              fontsize=9, color="0.45", va="bottom")

#     # t = time.time()
#     # fig.savefig(out, dpi=dpi, facecolor="white")
#     # print(f"[saved] {out} ({time.time()-t:.0f} s)")
#     # if show:
#     #     plt.show()
#     # else:
#     #     plt.close(fig)
#     # return fig


# def main(argv=None):
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--gpkg", default="conus_nextgen.gpkg")
#     ap.add_argument("--out", default="conus_hydrofabric_v2.2.png")
#     ap.add_argument("--dpi", type=int, default=300)
#     ap.add_argument("--simplify", type=float, default=150.0)
#     ap.add_argument("--min-order", type=int, default=3)
#     ap.add_argument("--no-gauges", action="store_true")
#     ap.add_argument("--no-inset", action="store_true")
#     args, _ = ap.parse_known_args(argv)
#     run_map(gpkg=args.gpkg, out=args.out, dpi=args.dpi, simplify=args.simplify,
#             min_order=args.min_order, show_gauges=not args.no_gauges,
#             inset=None if args.no_inset else (-71.6, -70.4, 44.4, 45.3), show=False)


# if __name__ == "__main__" and "ipykernel" not in sys.modules:
#     main()

# run_map()