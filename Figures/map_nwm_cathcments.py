"""
CONUS map of the catchments used by the operational National Water Model (v3.x):
NHDPlusV2.1 medium-resolution catchments and network flowlines, styled to match
map_conus_hydrofabric.py. Optional zoom inset overlays NextGen v2.2 divide
boundaries on the NHDPlus catchments for direct comparison.

Jupyter:
    run_nwm_map()                                         # downloads NHDPlusV2 if needed
    run_nwm_map(gdb="/path/NHDPlusV21_National_Seamless_Flattened_Lower48.gdb")

Requirements:
    pip install geopandas pyogrio pyarrow shapely matplotlib cartopy pyproj requests py7zr
    (a system 7z / 7za / 7zz binary is used instead of py7zr when available; much faster)
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import requests
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pyproj import Transformer
from shapely import get_coordinates, get_parts

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
NHDPLUS_URL = ("https://dmap-data-commons-ow.s3.amazonaws.com/NHDPlusV21/Data/NationalData/"
               "NHDPlusV21_NationalData_Seamless_Geodatabase_Lower48_07.7z")
NGIAB_GPKG = os.path.expanduser("~/.ngiab/hydrofabric/v2.2/conus_nextgen.gpkg")
CRS = "EPSG:5070"

ALBERS = ccrs.AlbersEqualArea(central_longitude=-96.0, central_latitude=23.0,
                              false_easting=0.0, false_northing=0.0,
                              standard_parallels=(29.5, 45.5),
                              globe=ccrs.Globe(ellipse="GRS80"))

# ---------------------------------------------------------------------------
# Style (same basemap and palette as the NextGen map)
# ---------------------------------------------------------------------------
LAND = "whitesmoke"
OCEAN = "lightblue"
STATE_EDGE = "gray"
VPU_PALETTE = ["#e8d5b7", "#cfe3c3", "#f3d1c8", "#d9d2e9", "#fbe7a1", "#c8e0e8",
               "#e6cfe0", "#d6e6b8", "#f6dcb2", "#cbd8ea", "#efd0d0", "#d8e8d0",
               "#ead9c4", "#d0dcd8", "#f0e0c0", "#dcd0e8", "#cfe6dd", "#f2d6bd",
               "#d4dce8", "#e9e2b8", "#e0cfc8"]
NO_VPU_COLOR = "#e4e4e4"
RIVER_CMAP = plt.get_cmap("Blues")
NEXTGEN_EDGE = "#b03a2e"
EXCLUDE_FTYPES = {"coastline", "pipeline"}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
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


def _find_gdb(root):
    hits = sorted(glob.glob(os.path.join(root, "**", "*.gdb"), recursive=True))
    return hits[0] if hits else None


def get_gdb(gdb=None, data_dir="nhdplusv2"):
    """Return the NHDPlusV2.1 national seamless geodatabase path; download/extract if needed."""
    # A .gdb path given in either argument is used directly; never download in that case.
    for p in (gdb, data_dir):
        if p and os.path.normpath(p).lower().endswith(".gdb"):
            if os.path.isdir(p):
                print(f"[found] {p}")
                return p
            raise FileNotFoundError(f"Geodatabase not found: {p}")
    if gdb:
        raise FileNotFoundError(f"Geodatabase not found: {gdb}")
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    found = _find_gdb(data_dir)
    if found:
        print(f"[found] {found}")
        return found

    archive = os.path.join(data_dir, os.path.basename(NHDPLUS_URL))
    if not os.path.exists(archive):
        _stream(NHDPLUS_URL, archive)

    print(f"[extract] {archive}  (large; this takes a while)")
    exe = next((shutil.which(b) for b in ("7zz", "7z", "7za") if shutil.which(b)), None)
    if exe:
        subprocess.run([exe, "x", "-y", f"-o{data_dir}", archive], check=True,
                       stdout=subprocess.DEVNULL)
    else:
        import py7zr
        with py7zr.SevenZipFile(archive, "r") as z:
            z.extractall(path=data_dir)

    found = _find_gdb(data_dir)
    if not found:
        raise RuntimeError(f"No .gdb found after extracting {archive}")
    print(f"[ready] {found}")
    return found


def pick_layer(layers, *candidates):
    names = {n.lower(): n for n in layers}
    for c in candidates:
        if c.lower() in names:
            return names[c.lower()]
    return None


def pick_field(fields, *candidates):
    names = {f.lower(): f for f in fields}
    for c in candidates:
        if c.lower() in names:
            return names[c.lower()]
    return None


def read(src, layer, columns=None):
    t = time.time()
    df = gpd.read_file(src, layer=layer, engine="pyogrio", use_arrow=True, columns=columns)
    if df.crs is None:
        df = df.set_crs("EPSG:4269")
    if df.crs.to_string() != CRS:
        df = df.to_crs(CRS)
    print(f"[read] {layer}: {len(df):,} features ({time.time()-t:.0f} s)")
    return df


# ---------------------------------------------------------------------------
# Geometry -> matplotlib collections
# ---------------------------------------------------------------------------
def polygon_rings(geoms):
    rings = []
    for poly in get_parts(geoms):
        if poly is None or poly.is_empty:
            continue
        rings.append(np.asarray(poly.exterior.coords)[:, :2])
    return rings


def line_segments(geoms):
    return [get_coordinates(g)[:, :2] for g in get_parts(geoms)
            if g is not None and not g.is_empty]


def river_style(order, o_min, o_max, inset=False):
    f = 0.0 if o_max == o_min else (order - o_min) / (o_max - o_min)
    color = RIVER_CMAP(0.45 + 0.55 * f)
    width = (0.25 + 1.6 * f ** 1.4) if inset else (0.08 + 1.1 * f ** 1.6)
    return color, width


def _load_features(detail):
    feats = {
        "ocean": cfeature.OCEAN.with_scale(detail),
        "land": cfeature.LAND.with_scale(detail),
        "lakes": cfeature.LAKES.with_scale(detail),
        "borders": cfeature.BORDERS.with_scale(detail),
        "coast": cfeature.COASTLINE.with_scale(detail),
        "states": cfeature.STATES.with_scale(detail),
    }
    for f in feats.values():
        next(iter(f.geometries()))
    return feats


def add_basemap(ax, detail="50m"):
    feats = None
    for d in dict.fromkeys([detail, "110m"]):
        try:
            feats = _load_features(d)
            break
        except Exception as e:
            print(f"[warn] Natural Earth {d} unavailable: {e}")
    if feats is None:
        ax.set_facecolor(LAND)
        return False
    ax.add_feature(feats["ocean"], facecolor=OCEAN, edgecolor="none", zorder=0)
    ax.add_feature(feats["land"], facecolor=LAND, edgecolor="none", zorder=0)
    ax.add_feature(feats["lakes"], facecolor=OCEAN, edgecolor="none", zorder=3)
    ax.add_feature(feats["borders"], edgecolor="0.35", linewidth=0.6, zorder=6)
    ax.add_feature(feats["coast"], edgecolor="0.35", linewidth=0.6, zorder=6)
    ax.add_feature(feats["states"], edgecolor=STATE_EDGE, facecolor="none",
                   linewidth=0.5, zorder=6)
    return True


def draw_catchments(ax, cat, colors, edge, lw, alpha):
    for v, sub in cat.groupby("_vpu", dropna=False):
        fc = colors.get(str(v), NO_VPU_COLOR)
        ax.add_collection(PolyCollection(
            polygon_rings(sub.geometry.values), facecolor=fc, edgecolor=edge,
            linewidth=lw, alpha=alpha, transform=ax.transData, zorder=2))


def draw_rivers(ax, fl, order_col, o_min, o_max, inset=False):
    if order_col is None:
        ax.add_collection(LineCollection(line_segments(fl.geometry.values),
                                         colors=[RIVER_CMAP(0.75)], linewidths=0.2,
                                         transform=ax.transData, zorder=4))
        return
    orders = fl[order_col].clip(lower=o_min).astype(int)
    for o in sorted(orders.unique()):
        c, w = river_style(o, o_min, o_max, inset)
        ax.add_collection(LineCollection(line_segments(fl.geometry[orders == o].values),
                                         colors=[c], linewidths=w, capstyle="round",
                                         joinstyle="round", transform=ax.transData,
                                         zorder=4))


# ---------------------------------------------------------------------------
# Main map
# ---------------------------------------------------------------------------
def run_nwm_map(gdb=None, data_dir="nhdplusv2", out="conus_nwm_nhdplusv2_catchments.png",
                dpi=300, simplify=150.0, min_order=3,
                inset=(-71.6, -70.4, 44.4, 45.3), nextgen_gpkg=NGIAB_GPKG,
                basemap_detail="50m", show=True):
    """
    gdb            : NHDPlusV2.1 national seamless .gdb (downloaded to data_dir if None/missing)
    data_dir       : download/extract folder (needs ~25-30 GB free for archive + extraction)
    out            : output PNG path
    dpi            : output resolution
    simplify       : main-map simplification tolerance in metres (inset is unsimplified)
    min_order      : main-map flowlines with StreamOrde >= this (inset shows all orders)
    inset          : (lon_min, lon_max, lat_min, lat_max) zoom window, or None
    nextgen_gpkg   : NextGen v2.2 gpkg; its divide boundaries are overlaid in the inset
                     (set None to skip)
    basemap_detail : Natural Earth scale "110m", "50m" or "10m"
    show           : display inline (Jupyter)
    """
    src = get_gdb(gdb, data_dir)
    layers = [l[0] for l in pyogrio.list_layers(src)]
    lyr_cat = pick_layer(layers, "Catchment", "CatchmentSP", "catchments")
    lyr_fl = pick_layer(layers, "NHDFlowline_Network", "NHDFlowline", "flowlines")
    if lyr_cat is None or lyr_fl is None:
        raise RuntimeError(f"Catchment/flowline layers not found. Layers: {layers}")

    # --- flowlines (network)
    fl_fields = pyogrio.read_info(src, layer=lyr_fl)["fields"]
    comid_col = pick_field(fl_fields, "COMID")
    order_col = pick_field(fl_fields, "StreamOrde", "StreamOrder")
    vpu_col = pick_field(fl_fields, "VPUID", "VPU")
    ftype_col = pick_field(fl_fields, "FTYPE")
    fl_all = read(src, lyr_fl, columns=[c for c in (comid_col, order_col, vpu_col, ftype_col) if c])
    if ftype_col:
        fl_all = fl_all[~fl_all[ftype_col].astype(str).str.lower().isin(EXCLUDE_FTYPES)]
    if order_col:
        fl_all = fl_all[fl_all[order_col] > 0]
        o_vals = fl_all[order_col].astype(int)
        o_min_all, o_max = int(o_vals.min()), int(o_vals.max())
        fl = fl_all[fl_all[order_col] >= min_order].copy()
        print(f"[filter] main map: {order_col} >= {min_order}: {len(fl):,} flowlines")
    else:
        o_min_all = o_max = 1
        fl = fl_all.copy()

    # --- catchments
    cat_fields = pyogrio.read_info(src, layer=lyr_cat)["fields"]
    fid_col = pick_field(cat_fields, "FEATUREID", "FeatureID", "COMID")
    cat = read(src, lyr_cat, columns=[fid_col] if fid_col else [])

    # VPU for colouring: catchment FEATUREID == flowline COMID
    if fid_col and comid_col and vpu_col:
        lut = fl_all[[comid_col, vpu_col]].drop_duplicates(comid_col).set_index(comid_col)[vpu_col]
        cat["_vpu"] = cat[fid_col].map(lut).astype("string")
        vpus = sorted(cat["_vpu"].dropna().unique())
        colors = {v: VPU_PALETTE[i % len(VPU_PALETTE)] for i, v in enumerate(vpus)}
    else:
        cat["_vpu"] = pd.NA
        colors = {}

    # --- inset subsets (unsimplified)
    box = ng_in = None
    if inset is not None:
        tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
        lon0, lon1, lat0, lat1 = inset
        xs, ys = tr.transform([lon0, lon1, lon0, lon1], [lat0, lat0, lat1, lat1])
        box = (min(xs), max(xs), min(ys), max(ys))
        cat_in = cat.cx[box[0]:box[1], box[2]:box[3]]
        fl_in = fl_all.cx[box[0]:box[1], box[2]:box[3]]
        if nextgen_gpkg and os.path.exists(nextgen_gpkg):
            ng_in = gpd.read_file(nextgen_gpkg, layer="divides", engine="pyogrio",
                                  bbox=(box[0], box[2], box[1], box[3]), columns=[])
            ng_in = ng_in.to_crs(CRS) if ng_in.crs and ng_in.crs.to_string() != CRS else ng_in
            print(f"[inset] NextGen divides in window: {len(ng_in):,}")

    t = time.time()
    if simplify > 0:
        cat["geometry"] = cat.geometry.simplify(simplify, preserve_topology=True)
        fl["geometry"] = fl.geometry.simplify(simplify, preserve_topology=False)
        print(f"[simplify] {time.time()-t:.0f} s")

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(18, 10.5), facecolor="white")
    ax = fig.add_axes([0.01, 0.06, 0.74, 0.86], projection=ALBERS)
    xmin, ymin, xmax, ymax = cat.total_bounds
    pad = 0.015 * (xmax - xmin)
    ax.set_extent([xmin - pad, xmax + pad, ymin - pad, ymax + pad], crs=ALBERS)
    add_basemap(ax, basemap_detail)

    t = time.time()
    draw_catchments(ax, cat, colors, edge=(1, 1, 1, 0.30), lw=0.03, alpha=0.95)
    draw_rivers(ax, fl, order_col, min_order, o_max)
    print(f"[draw] main map {time.time()-t:.0f} s")
    ax.spines["geo"].set_edgecolor("0.6")
    if box is not None:
        ax.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                               fill=False, edgecolor="black", linewidth=1.3,
                               transform=ALBERS, zorder=8))

    fig.text(0.015, 0.955, "National Water Model (v3.x) Catchments — NHDPlusV2.1, CONUS",
             fontsize=21, weight="bold", color="0.12")
    fig.text(0.015, 0.925,
             f"{len(cat):,} NHDPlusV2 catchments  ·  {len(fl_all):,} network flowlines "
             f"(NWM channel reaches)",
             fontsize=12.5, color="0.35")

    # ------------------------------------------------------------------ inset
    if box is not None:
        axi = fig.add_axes([0.765, 0.43, 0.22, 0.46], projection=ALBERS)
        axi.set_extent([box[0], box[1], box[2], box[3]], crs=ALBERS)
        axi.set_facecolor(LAND)
        draw_catchments(axi, cat_in, colors, edge="0.45", lw=0.3, alpha=1.0)
        draw_rivers(axi, fl_in, order_col, o_min_all, o_max, inset=True)
        if ng_in is not None and len(ng_in):
            axi.add_collection(PolyCollection(
                polygon_rings(ng_in.geometry.values), facecolor="none",
                edgecolor=NEXTGEN_EDGE, linewidth=1.1, transform=axi.transData, zorder=5))
        axi.spines["geo"].set_edgecolor("black")
        axi.spines["geo"].set_linewidth(1.3)
        title = f"Zoom: {len(cat_in):,} NHDPlusV2 catchments"
        if ng_in is not None:
            title += f"\nvs {len(ng_in):,} NextGen v2.2 divides (red)"
        axi.set_title(title, fontsize=11, color="0.25", loc="left", pad=6)
        sx = box[0] + 0.06 * (box[1] - box[0])
        sy = box[2] + 0.05 * (box[3] - box[2])
        axi.plot([sx, sx + 10_000], [sy, sy], color="black", lw=3, transform=ALBERS,
                 zorder=9, solid_capstyle="butt")
        axi.text(sx + 5_000, sy + 0.025 * (box[3] - box[2]), "10 km", ha="center",
                 fontsize=9, transform=ALBERS, zorder=9)

    # ------------------------------------------------------------------ legend
    handles = []
    if order_col:
        for o in sorted({min_order, (min_order + o_max) // 2, o_max}):
            c, w = river_style(o, min_order, o_max)
            handles.append(Line2D([], [], color=c, lw=max(w * 2.2, 0.8),
                                  label=f"Stream order {o}"))
    handles.append(Line2D([], [], color="none", marker="s", markersize=11,
                          markerfacecolor=VPU_PALETTE[0], markeredgecolor="0.4",
                          label="NHDPlusV2 catchments (by VPU)"))
    if ng_in is not None and len(ng_in):
        handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor=NEXTGEN_EDGE,
                                 linewidth=1.3, label="NextGen v2.2 divide (inset)"))
    if box is not None:
        handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black",
                                 linewidth=1.3, label="Zoom area"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.765, 0.40),
               frameon=False, fontsize=11.5, handlelength=2.6, labelspacing=0.9,
               title="Legend", title_fontproperties={"weight": "bold", "size": 12.5},
               alignment="left")

    fig.text(0.765, 0.07,
             "Catchments/flowlines: NHDPlusV2.1 (EPA/USGS),\n"
             "basis of the NWM v3 channel network\n"
             "Basemap: Natural Earth  ·  Projection: EPSG:5070",
             fontsize=9, color="0.45", va="bottom")

    t = time.time()
    fig.savefig(out, dpi=dpi, facecolor="white")
    print(f"[saved] {out} ({time.time()-t:.0f} s)")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdb", default=None)
    ap.add_argument("--data-dir", default="nhdplusv2")
    ap.add_argument("--out", default="conus_nwm_nhdplusv2_catchments.png")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--simplify", type=float, default=150.0)
    ap.add_argument("--min-order", type=int, default=3)
    ap.add_argument("--no-inset", action="store_true")
    args, _ = ap.parse_known_args(argv)
    run_nwm_map(gdb=args.gdb, data_dir=args.data_dir, out=args.out, dpi=args.dpi,
                simplify=args.simplify, min_order=args.min_order,
                inset=None if args.no_inset else (-71.6, -70.4, 44.4, 45.3), show=False)


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()

run_nwm_map(data_dir="/bighome/hdagne1/NeuralNGEN/data/nhdplusv2/NHDPlusNationalData/NHDPlusV21_National_Seamless_Flattened_Lower48.gdb")