#!/bin/bash
#SBATCH --job-name=NextGen
#SBATCH --output=/bighome/hdagne1/NeuralNGEN/logs/ngen/ngen_%A_%a.out
#SBATCH --error=/bighome/hdagne1/NeuralNGEN/logs/ngen/ngen_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --array=0-1431%20

module load apptainer/1.3.4-1
source ~/.bashrc
conda activate nh-dev

N=/bighome/hdagne1/NeuralNGEN
G=/bighome/hdagne1/Data/CAMELS_US_ngiab_full
CONFIG_PATH="$N/ngen_run_config.yml"
MANIFEST_PATH="$N/manifests/ngen_run_manifest.txt"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST_PATH")
GAUGE_ID=$(echo "$LINE" | cut -d',' -f1)
SCENARIO=$(echo "$LINE" | cut -d',' -f2)
echo "Task $SLURM_ARRAY_TASK_ID -> gauge=$GAUGE_ID scenario=$SCENARIO  start=$(date)"

# Skip if a fresh t-route output (newer than the manifest) already exists.
OUT=$G/gage-$GAUGE_ID/outputs
if find "$OUT/troute_${SCENARIO}_stream" -name 'troute_output_*.nc' -newer "$MANIFEST_PATH" 2>/dev/null | grep -q .; then
    echo "already done (fresh t-route output exists) - skipping"
    exit 0
fi

# Clear any stale/partial outputs for this gauge+scenario before running.
rm -f "$OUT/ngen_${SCENARIO}"/* "$OUT/troute_${SCENARIO}"/* "$OUT/troute_${SCENARIO}_stream"/* 2>/dev/null
rm -rf /scratch/hdagne1/qlat_cache/gage-${GAUGE_ID}/${SCENARIO}/* 2>/dev/null

cd "$N"
python3 "$N/run_ngen_scenario.py" "$CONFIG_PATH" "$GAUGE_ID" "$SCENARIO"
RC=$?
echo "Task done rc=$RC end=$(date)"
exit $RC














































# #!/bin/bash
# #SBATCH --job-name=NextGen
# #SBATCH --output=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/ngen/ngen_%A_%a.out
# #SBATCH --error=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/ngen/ngen_%A_%a.err
# #SBATCH --partition=main
# #SBATCH --qos=main
# #SBATCH --time=00:30:00
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=16G
# #SBATCH --array=0-1073%20

# module load apptainer/1.3.4-1
# source ~/.bashrc
# conda activate nh-dev

# CONFIG_PATH="/bighome/hdagne1/Docker/NeuralNGEN/ngen_run_config.yml"
# MANIFEST_PATH="/bighome/hdagne1/Docker/NeuralNGEN/manifests/ngen_run_manifest.txt"

# LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
# LINE=$(sed -n "${LINE_NUM}p" "$MANIFEST_PATH")

# GAUGE_ID=$(echo "$LINE" | cut -d',' -f1)
# SCENARIO=$(echo "$LINE" | cut -d',' -f2)

# echo "Task $SLURM_ARRAY_TASK_ID -> gauge=$GAUGE_ID scenario=$SCENARIO"

# python3 /bighome/hdagne1/Docker/NeuralNGEN/run_ngen_scenario.py \
#     "$CONFIG_PATH" "$GAUGE_ID" "$SCENARIO"











# #!/bin/bash
# #SBATCH --job-name=regen_ngen_configs
# #SBATCH --output=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/ngen/regen_%j.out
# #SBATCH --error=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/ngen/regen_%j.err
# #SBATCH --partition=main
# #SBATCH --qos=main
# #SBATCH --time=02:00:00
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=16G

# source ~/.bashrc
# conda activate nh-dev

# cd /bighome/hdagne1/Docker/NeuralNGEN


# echo "=== Step 1: regenerate realization/troute configs ==="
# python3 scripts/build_scaled_configs.py \
#     /bighome/hdagne1/Docker/Data/CAMELS_US_ngiab_full \
#     /bighome/hdagne1/Docker/NeuralNGEN/bmi_config_files \
#     fdc,rfl,both \
#     /bighome/hdagne1/Docker/ngen/data/gauge_01073000/routing_config.yaml

# echo "=== Step 2: rebuild manifest ==="
# python3 manifests/generate_manisfest.py \
#     /bighome/hdagne1/Docker/Data/CAMELS_US_ngiab_full \
#     /bighome/hdagne1/Docker/NeuralNGEN/manifests/ngen_run_manifest.txt

# echo "=== All steps complete ==="
# echo "IMPORTANT: check the 'Wrote N (gauge, scenario) lines' and '--array' line"
# echo "printed just above - if N differs from what run_ngen_array.sh's"
# echo "#SBATCH --array=0-N-1%20 currently says, update that line before submitting."