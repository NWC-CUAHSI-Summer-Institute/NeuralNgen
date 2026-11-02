#!/usr/bin/env python3
"""
filter_streamflow_lstm.py
===========================
LSTM-oriented QA/QC and cleaning pass over the daily USGS streamflow
parquet cache (datetime index, `flow_m3s` column), implementing the
criteria that are actually computable from what's in that cache today
(no USGS `cd` quality flags, no GAGES-II regulation class, no drainage
area are present in this data - those need separate sources; see the
script header note at the bottom for what's NOT covered).

Criteria applied, per basin:

  1. Per-year missing-data threshold (not just whole-record): any
     calendar year with more than --max-missing-pct-per-year missing
     days is flagged. This matters because LSTM look-back windows
     (weeks to a year) can sit entirely inside one bad year even if
     the basin's overall record looks fine.

  2. Short-gap interpolation: gaps of --max-interp-gap-days or fewer
     are linearly interpolated (these are written to --out-dir if
     given - the source cache is never modified in place). Gaps
     longer than that are left as real gaps (NaN), not papered over.

  3. Zero/negative flow: negative values are a hard integrity fail
     (physically impossible for this cache's units). Zero-flow
     frequency is reported, not auto-failed - only matters if you
     later log-transform, and that's your modeling call, not a
     universal QC rule.

  4. Day-to-day physical-consistency check: flags days where the
     log-ratio jump |log(Q_t) - log(Q_t-1)| is an extreme outlier
     relative to that SAME basin's own historical jump distribution
     (median + k * MAD, robust to the basin's own flashiness - a
     flashy mountain basin and a slow lowland basin get different
     effective thresholds). This is a proxy for sensor glitches, not
     a precipitation cross-check (no forcing data used here) - it
     flags for review, it does not drop days automatically.

PASS/FAIL: a basin fails only on real integrity problems (negative
values, or a year exceeding the missing-data threshold). Zero-flow
frequency and flagged jumps are reported, not disqualifying, since
those are modeling-choice-dependent, not universal QC failures.

GAGES-II (if --gages-ii is given): adds `CLASS` (Ref/Non-ref) and
`DRAIN_SQKM` to the report for every basin GAGES-II covers (it's a
~2400-site subset, not the full national inventory - basins outside it
show up as NaN/"Not in GAGES-II", which is expected, not a failure).
  - --require-reference: fail any basin that is NOT "Ref" (or not in
    GAGES-II at all, since regulation status is then unknown). Off by
    default - this is a real modeling choice (natural-flow-only vs.
    including regulated basins), not a universal QC rule.
  - --require-drainage-area: fail any basin with missing/zero
    DRAIN_SQKM (needed if you'll normalize flow to runoff depth,
    mm/day). Off by default for the same reason.

NOT covered here (need data this cache/GAGES-II doesn't have):
  - USGS qualification/quality codes (A/P/E/Ice) - would need the
    `..._cd` column from the original nwis.get_dv() pull, which this
    cache doesn't retain.

Usage
-----
    python filter_streamflow_lstm.py \\
        --cache-dir /bighome/hdagne1/usgs-streamflow-inventory/data/raw/usgs_daily_cache \\
        --gages-ii /bighome/hdagne1/conus-lowflow-hydro-drought-thresholds/data/gage-II/gages_ii.parquet \\
        --out-csv lstm_qc_report.csv \\
        --out-dir /bighome/hdagne1/usgs-streamflow-inventory/data/processed/usgs_daily_cleaned \\
        --max-missing-pct-per-year 5 --max-interp-gap-days 3 --jump-mad-k 10
        # add --require-reference and/or --require-drainage-area to enforce those
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_site(cache_dir: Path, gauge: str) -> pd.DataFrame | None:
    path = cache_dir / f"{gauge}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            return None
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    if "flow_m3s" not in df.columns:
        return None
    return df[["flow_m3s"]].sort_index()


def reindex_full_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto a continuous daily range over the basin's own record so
    missing days show up as real NaN rows, not just absent rows."""
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(full_range)


def per_year_missing(df: pd.DataFrame) -> pd.DataFrame:
    s = df["flow_m3s"]
    by_year = s.groupby(s.index.year)
    total = by_year.size()
    missing = by_year.apply(lambda x: x.isna().sum())
    pct = 100.0 * missing / total
    return pd.DataFrame({"n_days": total, "n_missing": missing, "pct_missing": pct})


def interpolate_short_gaps(df: pd.DataFrame, max_gap_days: int) -> tuple[pd.DataFrame, int]:
    """Linearly interpolate runs of NaN up to max_gap_days long. Longer runs
    are left as NaN. Returns (new_df, n_days_interpolated)."""
    s = df["flow_m3s"].copy()
    is_na = s.isna()
    if not is_na.any():
        return df, 0

    run_id = (is_na != is_na.shift()).cumsum()
    run_lengths = is_na.groupby(run_id).transform("sum")
    fillable = is_na & (run_lengths <= max_gap_days)

    s_interp = s.interpolate(method="linear", limit_area="inside")
    s_new = s.copy()
    s_new[fillable] = s_interp[fillable]

    n_filled = int(fillable.sum())
    out = df.copy()
    out["flow_m3s"] = s_new
    return out, n_filled


def flag_extreme_jumps(df: pd.DataFrame, mad_k: float) -> int:
    """Flag days whose log-ratio day-over-day jump is an extreme outlier
    relative to THIS basin's own jump distribution (robust median+MAD)."""
    s = df["flow_m3s"]
    positive = s.where(s > 0)  # log undefined at/below 0; those are handled separately
    log_q = np.log(positive)
    diffs = log_q.diff().dropna()
    if len(diffs) < 30:
        return 0
    med = diffs.median()
    mad = (diffs - med).abs().median()
    if mad == 0:
        return 0
    robust_z = (diffs - med).abs() / (1.4826 * mad)
    return int((robust_z > mad_k).sum())


def load_gages_ii(path: Path) -> dict:
    """Return {8-digit gauge id -> (CLASS, DRAIN_SQKM)}."""
    g = pd.read_parquet(path)
    g["site_no"] = g["site_no"].astype(str).str.zfill(8)
    return {
        row.site_no: (row.CLASS, row.DRAIN_SQKM)
        for row in g[["site_no", "CLASS", "DRAIN_SQKM"]].itertuples()
    }


def qc_one_basin(gauge: str, raw_df: pd.DataFrame, max_missing_pct_per_year: float,
                  max_interp_gap_days: int, jump_mad_k: float,
                  gages_ii: dict | None, require_reference: bool,
                  require_drainage_area: bool) -> tuple[dict, pd.DataFrame]:
    df = reindex_full_daily(raw_df)

    n_negative = int((df["flow_m3s"] < 0).sum())
    n_zero = int((df["flow_m3s"] == 0).sum())
    n_total = int(df["flow_m3s"].notna().sum())
    pct_zero = 100.0 * n_zero / n_total if n_total else np.nan

    yearly = per_year_missing(df)
    n_years = len(yearly)
    n_bad_years = int((yearly["pct_missing"] > max_missing_pct_per_year).sum())
    worst_year_pct = float(yearly["pct_missing"].max()) if n_years else np.nan

    df_clean, n_interpolated = interpolate_short_gaps(df, max_interp_gap_days)

    n_extreme_jumps = flag_extreme_jumps(df_clean, jump_mad_k)

    record_start, record_end = df.index.min(), df.index.max()

    gages_class, drain_sqkm = (None, np.nan)
    if gages_ii is not None and gauge in gages_ii:
        gages_class, drain_sqkm = gages_ii[gauge]

    passed = (n_negative == 0) and (n_bad_years == 0)
    if require_reference:
        passed = passed and (gages_class == "Ref")
    if require_drainage_area:
        passed = passed and (drain_sqkm == drain_sqkm) and (drain_sqkm > 0)  # not NaN and > 0

    summary = dict(
        gauge=gauge,
        record_start=record_start,
        record_end=record_end,
        n_years=n_years,
        n_bad_years_over_threshold=n_bad_years,
        worst_year_pct_missing=round(worst_year_pct, 2) if worst_year_pct == worst_year_pct else np.nan,
        n_negative_values=n_negative,
        pct_zero_flow=round(pct_zero, 3) if pct_zero == pct_zero else np.nan,
        n_days_interpolated=n_interpolated,
        n_extreme_jumps_flagged=n_extreme_jumps,
        gages_class=gages_class if gages_class is not None else "Not in GAGES-II",
        drain_sqkm=drain_sqkm,
        PASS=passed,
    )
    return summary, df_clean


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--basin-list", type=Path, default=None,
                     help="One gauge ID per line. If omitted, every *.parquet under --cache-dir is used.")
    ap.add_argument("--out-csv", type=Path, default=Path("lstm_qc_report.csv"))
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="If given, writes cleaned (gap-interpolated) parquet files here for basins "
                          "that PASS, one file per basin, same filename as the source. Source cache "
                          "is never modified.")
    ap.add_argument("--max-missing-pct-per-year", type=float, default=5.0)
    ap.add_argument("--max-interp-gap-days", type=int, default=3)
    ap.add_argument("--jump-mad-k", type=float, default=10.0,
                     help="Robust z-score threshold (median + k*MAD) on log-ratio day-to-day jumps, "
                          "computed per-basin against its own jump distribution.")
    ap.add_argument("--gages-ii", type=Path, default=None,
                     help="Path to gages_ii.parquet (columns: site_no, CLASS, DRAIN_SQKM, ...). "
                          "If given, adds gages_class/drain_sqkm to the report.")
    ap.add_argument("--require-reference", action="store_true",
                     help="Fail any basin that is not GAGES-II CLASS=='Ref' (needs --gages-ii).")
    ap.add_argument("--require-drainage-area", action="store_true",
                     help="Fail any basin with missing/zero DRAIN_SQKM (needs --gages-ii).")
    args = ap.parse_args()

    if (args.require_reference or args.require_drainage_area) and args.gages_ii is None:
        ap.error("--require-reference / --require-drainage-area need --gages-ii")

    gages_ii = load_gages_ii(args.gages_ii) if args.gages_ii is not None else None

    if args.basin_list is not None:
        gauges = [line.strip().zfill(8) for line in args.basin_list.read_text().splitlines() if line.strip()]
    else:
        gauges = sorted(p.stem for p in args.cache_dir.glob("*.parquet"))
        print(f"No --basin-list given: checking all {len(gauges)} *.parquet files under {args.cache_dir}")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_files = []
    for gauge in gauges:
        raw = load_site(args.cache_dir, gauge)
        if raw is None:
            missing_files.append(gauge)
            continue
        summary, cleaned = qc_one_basin(gauge, raw, args.max_missing_pct_per_year,
                                         args.max_interp_gap_days, args.jump_mad_k,
                                         gages_ii, args.require_reference, args.require_drainage_area)
        rows.append(summary)
        if args.out_dir is not None and summary["PASS"]:
            cleaned.to_parquet(args.out_dir / f"{gauge}.parquet")

    report = pd.DataFrame(rows)
    report.to_csv(args.out_csv, index=False)

    n_total, n_found = len(gauges), len(rows)
    n_pass = int(report["PASS"].sum()) if len(report) else 0

    print(f"Basins requested:      {n_total}")
    print(f"Cache files found:     {n_found}")
    print(f"Cache files MISSING:   {len(missing_files)}")
    print(f"Passed:                {n_pass} / {n_found}")
    print(f"Failed (negative vals or a bad year): {n_found - n_pass} / {n_found}")
    print(f"Report written to:     {args.out_csv}")
    if args.out_dir is not None:
        print(f"Cleaned parquet files written to: {args.out_dir} (PASS basins only)")

    if len(report):
        fail_path = args.out_csv.with_name(args.out_csv.stem + "_failed_gauges.txt")
        report.loc[~report["PASS"], "gauge"].to_csv(fail_path, index=False, header=False)
        print(f"Failed gauge list:     {fail_path}")


if __name__ == "__main__":
    main()