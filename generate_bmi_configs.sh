#!/usr/bin/env bash
#SBATCH --job-name=NGIAB
#SBATCH --output=/bighome/hdagne1/NeuralNGEN/runs/logs/gen_bmi_configs/gen_%A_%a.out
#SBATCH --error=/bighome/hdagne1/NeuralNGEN/runs/logs/gen_bmi_configs/gen_%A_%a.err
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-439%40

source ~/.bashrc
conda activate nh-dev

MANIFEST_PATH="/bighome/hdagne1/NeuralNGEN/basin_lists/ready_basins_filtered_updated.txt"
RUNS_ROOT="/bighome/hdagne1/NeuralNGEN/runs"
GAUGES_ROOT="/bighome/hdagne1/Data/CAMELS_US_ngiab_full"
NEURALNGEN_ROOT="/bighome/hdagne1/NeuralNGEN"
BMI_CONFIG_ROOT="/bighome/hdagne1/NeuralNGEN/bmi_config_files"

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
GAUGE_ID=$(sed -n "${LINE_NUM}p" "$MANIFEST_PATH")

echo "Task $SLURM_ARRAY_TASK_ID -> gauge=$GAUGE_ID"

# Auto-discover every trained scenario instead of hardcoding one - any
# runs/scoring_rule_multi_<scenario>/model_final.pt on disk counts as a
# scenario to generate configs for. Add/remove checkpoints under runs/ and
# this picks them up automatically on the next submission, no editing here.
shopt -s nullglob
CHECKPOINTS=("$RUNS_ROOT"/scoring_rule_multi_*_v2/model_final.pt)

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: no checkpoints found matching $RUNS_ROOT/scoring_rule_multi_*/model_final.pt"
    exit 1
fi

echo "Found ${#CHECKPOINTS[@]} scenario checkpoint(s): ${CHECKPOINTS[*]}"

EXIT_CODE=0
for CKPT in "${CHECKPOINTS[@]}"; do
    RUN_DIR=$(dirname "$CKPT")
    SCENARIO=$(basename "$RUN_DIR" | sed 's/^scoring_rule_multi_//')

    echo "--- gauge=$GAUGE_ID scenario=$SCENARIO checkpoint=$CKPT ---"
    python3 /bighome/hdagne1/NeuralNGEN/bmi_config_files/generate_bmi_configs.py \
        "$GAUGE_ID" "$SCENARIO" "$CKPT" \
        "$GAUGES_ROOT" "$NEURALNGEN_ROOT" "$BMI_CONFIG_ROOT"
    RC=$?

    if [ $RC -ne 0 ]; then
        echo "  gauge=$GAUGE_ID scenario=$SCENARIO FAILED (exit $RC) - likely means this "
        echo "  gauge wasn't part of the $SCENARIO checkpoint's training set; continuing "
        echo "  with remaining scenarios for this gauge."
        EXIT_CODE=1
    fi
done

exit $EXIT_CODE
