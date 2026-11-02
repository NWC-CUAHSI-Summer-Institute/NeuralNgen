#!/usr/bin/env bash
#SBATCH --job-name=NGIAB
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ngiab_%A_%a.out
#SBATCH --error=logs/ngiab_%A_%a.err

set -o pipefail

BASIN_LIST=/bighome/hdagne1/Docker/Data/basin_list_still_missing_72.txt
OUTPUT_ROOT=/bighome/hdagne1/Docker/Data/CAMELS_US_ngiab_full
START_DATE=2000-01-01
END_DATE=2024-12-31

source /home/hdagne1/miniforge3/etc/profile.d/conda.sh
conda activate ngiab-dev

set -u

GAUGE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${BASIN_LIST}" | tr -d '[:space:]')
if [[ -z "${GAUGE}" ]]; then
    echo "No basin at array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

echo "[${GAUGE}] starting subset + forcings, ${START_DATE} to ${END_DATE}"

python -m ngiab_data_cli \
    -i "${GAUGE}" -g \
    -s -f \
    --start_date "${START_DATE}" --end_date "${END_DATE}" \
    --source aorc \
    --subset_type catchment \
    --output_root "${OUTPUT_ROOT}" \
    -o "gage-${GAUGE}"

status=$?

# Immediately delete this gauge's raw gridded file, regardless of success/
# failure, so disk usage never accumulates across the array - this is what
# blew the quota last time.
find "${OUTPUT_ROOT}/gage-${GAUGE}" -iname "*raw_gridded*.nc" -delete

if [[ ${status} -eq 0 ]]; then
    echo "[${GAUGE}] DONE"
else
    echo "[${GAUGE}] FAILED (exit ${status})" >&2
fi
exit ${status}