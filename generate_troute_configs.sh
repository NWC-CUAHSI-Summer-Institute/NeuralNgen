#!/usr/bin/env bash
#SBATCH --job-name=NGIAB_build_configs
#SBATCH --output=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/build_scaled_configs/build_%j.out
#SBATCH --error=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/build_scaled_configs/build_%j.err
#SBATCH --partition=main
#SBATCH --qos=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

source activate nh-dev

python3 /bighome/hdagne1/Docker/NeuralNGEN/scripts/build_scaled_configs.py \
    /bighome/hdagne1/Docker/Data/CAMELS_US_ngiab_full \
    /bighome/hdagne1/Docker/NeuralNGEN/bmi_config_files \
    all,fdc,rfl \
    /bighome/hdagne1/Docker/ngen/data/gauge_01073000/routing_config.yaml