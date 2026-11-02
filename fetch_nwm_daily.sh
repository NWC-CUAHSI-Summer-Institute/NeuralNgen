#!/bin/bash
#SBATCH --job-name=nwm_daily
#SBATCH --output=/bighome/hdagne1/NeuralNGEN/logs/nwm_daily_%j.out
#SBATCH --error=/bighome/hdagne1/NeuralNGEN/logs/nwm_daily_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source ~/.bashrc
conda activate nh-dev

export WORKER_MEM=3.5GB                  # 64G / 16 workers
python /bighome/hdagne1/NeuralNGEN/scripts/fetch_nwm_daily.py