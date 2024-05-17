#!/bin/bash

# Usage: sbatch scripts/multi_node_training.sh

#SBATCH --job-name  train
#SBATCH --time      UNLIMITED
#SBATCH --nodes             1
#SBATCH --ntasks-per-node   1
#SBATCH --output    %x-%J.out
#SBATCH --gres          gpu:2

CONDA_HOME="${HOME}/conda"
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate deepfold2-dev
BASE="/runs/users/vv137/project/DeepFold"

export PYTHONPATH=$BASE

# Print current datetime:
echo "START" $(date +"%Y-%m-%d %H:%M:%S")

if [[ -n "$SLURM_JOB_ID" ]]; then
    # Print node list:
    echo "SLURM_JOB_ID=$SLURM_JOB_ID"
    echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
    echo "SLURM_NODELIST=$SLURM_NODELIST"

    NNODES=$SLURM_JOB_NUM_NODES
    JOB_ID=$SLURM_JOB_ID

    export MASTER_ADDR=$(echo $SLURM_NODELIST | cut -d ',' -f 1)
    export MASTER_PORT=$SLURM_JOB_ID

else
    NNODES=1
    JOB_ID=10001

    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=10001

fi

# Set number of threads to use for parallel regions:
export OMP_NUM_THREADS=16

# PyTorch Dynamo
export TORCH_COMPILE_DISABLE=1

INPUT_FEAT=$1
OUTPUT_DIR=$2
shift 2

# srun --export=ALL \
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=2 \
    --rdzv_id=$JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    predict.py \
    --mp_size 2 \
    --input_features_filepath $INPUT_FEAT \
    --output_dirpath $OUTPUT_DIR \
    --params_dirpath "/gpfs/database/casp16/params" \
    "$@"
# --preset "params_model_1" \

echo "END" $(date +"%Y-%m-%d %H:%M:%S")
