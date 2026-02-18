#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cluster=htc
#SBATCH --partition=htc
# SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --mail-user=aaf92@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ============================================
# Ensure logs folder exists (for SLURM logs)
# ============================================

mkdir -p logs

# ============================================
# Config
# ============================================

CONFIG_PATH="/ix/djishnu/Aaron_F/ES_interact/Code/substrate_attention/configs/attention_v2.yaml"

# Extract save_dir from YAML
SAVE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG_PATH}'))['experiment']['save_dir'])")

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="run_${TIMESTAMP}_job${SLURM_JOB_ID}"
RUN_DIR="${SAVE_DIR}/${RUN_NAME}"

mkdir -p ${RUN_DIR}

# ============================================
# Redirect Python output to run directory
# ============================================

exec > >(tee -a ${RUN_DIR}/train.log)
exec 2> >(tee -a ${RUN_DIR}/train.err >&2)

echo "========================================="
echo "Run Directory: ${RUN_DIR}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "========================================="

# ============================================
# Environment
# ============================================

module load python/ondemand-jupyter-python3.11
module load cuda/12.9.0
source activate ES_interact_env

# ============================================
# Run Training
# ============================================

python /ix/djishnu/Aaron_F/ES_interact/Code/substrate_attention/main.py \
    --config ${CONFIG_PATH} \
    --run_dir ${RUN_DIR}

echo "Finished at: $(date)"
