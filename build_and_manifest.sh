#!/bin/bash
source ~/.bashrc
conda activate nh-dev
cd /bighome/hdagne1/NeuralNGEN
G=/bighome/hdagne1/Data/CAMELS_US_ngiab_full
N=/bighome/hdagne1/NeuralNGEN
T=/bighome/hdagne1/Docker/ngen/data/gauge_01073000/routing_config.yaml

echo "window in script:"; grep -n "^START_TIME\|^END_TIME\|^TROUTE_START\|^TROUTE_NTS" scripts/build_scaled_configs.py

for sc in all fdc rfl mse; do
  python3 scripts/build_scaled_configs.py $G $N/bmi_config_files $sc $T > runs/logs/build_scn_$sc.log 2>&1 &
done
wait

python3 manifests/generate_manisfest.py $G $N/manifests/ngen_run_manifest.txt

echo "--- T-ROUTE TIME KEYS (generated) ---"
grep -n "start_datetime\|nts\|max_loop_size\|^ *dt:\|qts_subdivisions" $G/gage-01047000/config/mse/troute_mse.yaml
echo BUILD_DONE
