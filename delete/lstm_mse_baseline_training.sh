#!/bin/bash
#SBATCH --job-name=NexGenMSE
#SBATCH --output=/bighome/hdagne1/NeuralNGEN/runs/logs/lstmMSE_%j.out
#SBATCH --error=/bighome/hdagne1/NeuralNGEN/runs/logs/lstmMSE_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate nh-dev

cd /bighome/hdagne1/NeuralNGEN
export PYTHONPATH=/bighome/hdagne1:${PYTHONPATH}

echo "scenario=mse (baseline: pointwise MSE only)"
python -u main.py train_multi --config configs/scoring_rule_multi_mse.yml --gpu 0
python -u main.py evaluate_multi --config configs/scoring_rule_multi_mse.yml --gpu 0
