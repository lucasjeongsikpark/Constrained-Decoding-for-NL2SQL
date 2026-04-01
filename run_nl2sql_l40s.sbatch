#!/bin/bash
#SBATCH --job-name=nl2sql
#SBATCH --partition=qcbgpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ======================================
# Create log directory if not exists
# ======================================
mkdir -p logs

# ======================================
# Parse input arguments
# Usage:
# sbatch run_nl2sql_l40s.sbatch [dataset] [constraint] [mode]
# ======================================
DATASET=${1:-wikisql}
CONSTRAINT=${2:-outlines}
MODE=${3:-non-thinking}

# ======================================
# Define experiment tag (used for logging/output naming)
# ======================================
EXP_NAME=${DATASET}_${CONSTRAINT}_${MODE}

echo "===== Job Configuration ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Dataset: $DATASET"
echo "Constraint: $CONSTRAINT"
echo "Mode: $MODE"
echo "Experiment Name: $EXP_NAME"
echo "Start time: $(date)"
echo "============================="

# ======================================
# Move to project directory
# ======================================
cd /scratch1/jeongsik/CSCI566-GroupProject || exit 1

# ======================================
# Load required modules
# ======================================
module purge
module load python/3.10

# ======================================
# Activate conda environment
# ======================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nl2sql

# ======================================
# Set Hugging Face cache directory (avoid home quota issues)
# ======================================
export HF_HOME=/scratch1/jeongsik/.cache/huggingface
mkdir -p "$HF_HOME"

# ======================================
# Print GPU info (sanity check)
# ======================================
echo "===== GPU Info ====="
nvidia-smi
echo "===================="

# ======================================
# Define output file path (optional)
# ======================================
OUTPUT_DIR=results
mkdir -p "$OUTPUT_DIR"

OUTPUT_FILE=${OUTPUT_DIR}/result_${EXP_NAME}.json

echo "Output will be saved to: $OUTPUT_FILE"

# ======================================
# Run experiment
# ======================================
python src/qwen35_hf_sql.py \
  --dataset "$DATASET" \
  --constraint "$CONSTRAINT" \
  --mode "$MODE" \
  --output_file "$OUTPUT_FILE"

# ======================================
# Job finished
# ======================================
echo "End time: $(date)"
echo "===== Job Finished ====="
