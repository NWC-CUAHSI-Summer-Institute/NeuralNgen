#!/bin/bash
#SBATCH --job-name=LSTM_eva
#SBATCH --output=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/eval_fdc_%j.out
#SBATCH --error=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/eval_fdc_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate nh-dev
cd /bighome/hdagne1/Docker/NeuralNGEN

python -u main.py evaluate_multi --config configs/scoring_rule_multi_fdc.yml --gpu 0















# #!/bin/bash
# #SBATCH --job-name=LSTM_eva
# #SBATCH --output=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/eval_%A_%a.out
# #SBATCH --error=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/eval_%A_%a.err
# #SBATCH --partition=gpu
# #SBATCH --qos=gpu
# #SBATCH --time=02:00:00
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=192G
# #SBATCH --gres=gpu:1
# #SBATCH --array=0-1

# source ~/.bashrc
# conda activate nh-dev
# cd /bighome/hdagne1/Docker/NeuralNGEN

# SCENARIOS=(rfl all)
# SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}

# echo "Evaluating scenario=$SCENARIO"
# python -u main.py evaluate_multi --config configs/scoring_rule_multi_${SCENARIO}.yml --gpu 0
