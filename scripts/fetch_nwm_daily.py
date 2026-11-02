import os
import numpy as np, pandas as pd, xarray as xr
from dask.distributed import Client, LocalCluster

BUCKET = "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr"
START, END = "2000-01-01", "2020-01-08"
OUT = "/bighome/hdagne1/NeuralNGEN/nwm_conus_daily"
FINAL = f"{OUT}/NWM_CONUS_Retrospect_daily.parquet"


def main():
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
    mem = os.environ.get("WORKER_MEM", "3.5GB")
    client = Client(LocalCluster(n_workers=n, threads_per_worker=1, memory_limit=mem))
    print(client.dashboard_link, flush=True)

    ds = xr.open_zarr(BUCKET, storage_options={"anon": True})
    gage = np.char.strip(ds["gage_id"].values.astype(str))
    idx = np.flatnonzero(gage != "")                  # gaged reaches only, chosen BEFORE any read
    gage = gage[idx]
    sf = ds["streamflow"].isel(feature_id=idx)
    print(f"{len(idx)} gaged reaches", flush=True)

    os.makedirs(OUT, exist_ok=True)
    for yr in range(int(START[:4]), int(END[:4]) + 1):
        f = f"{OUT}/daily_{yr}.parquet"
        if os.path.exists(f):
            continue
        t0, t1 = max(START, f"{yr}-01-01"), min(END, f"{yr}-12-31")
        d = sf.sel(time=slice(t0, t1)).resample(time="1D").mean().compute()
        pd.DataFrame(d.values, index=pd.DatetimeIndex(d["time"].values), columns=gage).to_parquet(f)
        print("saved", yr, flush=True)

    wide = pd.concat([pd.read_parquet(f"{OUT}/daily_{y}.parquet")
                      for y in range(int(START[:4]), int(END[:4]) + 1)]).sort_index()
    wide = wide.T.groupby(level=0).mean().T           # same as pivot_table(aggfunc="mean") for repeated gages
    wide.index.name = "Time"
    wide.to_parquet(FINAL)
    print("done:", wide.shape, FINAL, flush=True)


if __name__ == "__main__":
    main()