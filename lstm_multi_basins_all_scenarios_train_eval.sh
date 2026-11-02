#!/bin/bash
#SBATCH --job-name=LSTMgpu
#SBATCH --output=/bighome/hdagne1/NeuralNGEN/runs/logs/lstmgpu_%A_%a.out
#SBATCH --error=/bighome/hdagne1/NeuralNGEN/runs/logs/lstmgpu_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --requeue

source ~/.bashrc
conda activate nh-dev

cd /bighome/hdagne1/NeuralNGEN
export PYTHONPATH=/bighome/hdagne1:${PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs

VER=${VER:-v2}
SCENARIOS=(mse rfl fdc all)
SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}
CFG=configs/scoring_rule_multi_${SCENARIO}_${VER}.yml
RUN=runs/scoring_rule_multi_${SCENARIO}_${VER}

echo "Task $SLURM_ARRAY_TASK_ID -> scenario=$SCENARIO  version=$VER  config=$CFG"

if [ -f "$RUN/model_final.pt" ] && [ -z "$FORCE" ]; then
    echo "model_final.pt exists in $RUN -> skipping training (set FORCE=1 to retrain)"
else
    python -u main.py train_multi --config $CFG --gpu 0
fi

python -u main.py evaluate_multi --config $CFG --gpu 0


















# #!/bin/bash
# #SBATCH --job-name=LSTM4
# #SBATCH --output=/bighome/hdagne1/NeuralNGEN/runs/logs/lstm4_%A_%a.out
# #SBATCH --error=/bighome/hdagne1/NeuralNGEN/runs/logs/lstm4_%A_%a.err
# #SBATCH --partition=gpu
# #SBATCH --qos=gpu
# #SBATCH --time=72:00:00
# #SBATCH --cpus-per-task=32
# #SBATCH --mem=256G
# #SBATCH --gres=gpu:1
# #SBATCH --array=0-3
# #SBATCH --requeue

# source ~/.bashrc
# conda activate nh-dev

# cd /bighome/hdagne1/NeuralNGEN
# export PYTHONPATH=/bighome/hdagne1:${PYTHONPATH}
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# mkdir -p runs/logs

# SCENARIOS=(mse rfl fdc all)
# SCENARIO=${SCENARIOS[$SLURM_ARRAY_TASK_ID]}
# CFG=configs/scoring_rule_multi_${SCENARIO}_v2.yml
# RUN=runs/scoring_rule_multi_${SCENARIO}_v2

# echo "Task $SLURM_ARRAY_TASK_ID -> scenario=$SCENARIO  config=$CFG"

# if [ -f "$RUN/model_final.pt" ] && [ -z "$FORCE" ]; then
#     echo "model_final.pt exists in $RUN -> skipping training (set FORCE=1 to retrain)"
# else
#     python -u main.py train_multi --config $CFG --gpu 0
# fi

# python -u main.py evaluate_multi --config $CFG --gpu 0