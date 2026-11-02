"""
Generate ngen_run_manifest.txt by scanning gauges_root for gauges that
build_scaled_configs.py actually wrote configs for - one line per
(gauge_id, scenario) pair, "gauge_id,scenario", matching what
run_ngen_scenario.py / submit_ngen_array.sh expect.

Robust to whatever scenario names you used (rfl, fdc, both, ...) - it
infers them from the realization_<scenario>.json filenames actually
present, rather than hardcoding a scenario list.

Usage:
    python generate_manifest.py <gauges_root> <out_manifest_path>

Example:
    python generate_manifest.py \
        /bighome/hdagne1/Docker/NeuralNGEN/gauges_root \
        /bighome/hdagne1/Docker/NeuralNGEN/manifests/ngen_run_manifest.txt
"""
import re
import sys
from pathlib import Path

REALIZATION_RE = re.compile(r"^realization_(.+)\.json$")


def main():
    gauges_root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    lines = []
    gauge_dirs = sorted(gauges_root.glob("gage-*"))
    print(f"Scanning {len(gauge_dirs)} gauge directories under {gauges_root} ...")

    for gauge_dir in gauge_dirs:
        gauge_id = gauge_dir.name.replace("gage-", "")
        config_dir = gauge_dir / "config"
        if not config_dir.exists():
            continue
        for realization_file in sorted(config_dir.glob("realization_*.json")):
            m = REALIZATION_RE.match(realization_file.name)
            if not m:
                continue
            scenario = m.group(1)
            lines.append(f"{gauge_id},{scenario}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} (gauge, scenario) lines to {out_path}")
    print(f"\n--> set your sbatch script's --array to: 0-{len(lines) - 1}")
    if len(lines) != 1296:
        print(f"NOTE: this is {len(lines)}, not 1296 - update --array=0-1295 in "
              f"submit_ngen_array.sh to match (0-{len(lines) - 1}), or investigate "
              f"why the count differs from what you expected.")


if __name__ == "__main__":
    main()