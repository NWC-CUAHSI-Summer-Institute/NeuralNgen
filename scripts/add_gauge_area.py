import sqlite3, sys
from pathlib import Path
root = Path("/bighome/hdagne1/Data/CAMELS_US_ngiab_full")
pat = f"gage-{sys.argv[1]}" if len(sys.argv) > 1 else "gage-*"
n_g = n_f = 0
for gd in sorted(root.glob(pat)):
    gp = list((gd / "config").glob("*_subset.gpkg"))
    ymls = list(gd.glob("bmi_config_files_*/cat-*.yml"))
    if not gp or not ymls:
        continue
    con = sqlite3.connect(gp[0]); A = con.execute("SELECT SUM(areasqkm) FROM divides").fetchone()[0]; con.close()
    for y in ymls:
        t = y.read_text()
        if "gauge_area_sqkm" in t:
            continue
        if not t.endswith("\n"): t += "\n"
        y.write_text(t + f"gauge_area_sqkm: {A}\n"); n_f += 1
    n_g += 1
    print(gd.name, "done", flush=True)
print(f"gauges: {n_g}, configs updated: {n_f}")
