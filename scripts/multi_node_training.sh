#!/bin/bash -l

# Usage: sbatch scripts/multi_node_training.sh

#SBATCH --job-name=train
#SBATCH --time=UNLIMITED
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:4

CONDA_HOME="${HOME}/conda"
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate deepfold2-dev
BASE="/runs/users/vv137/project/DeepFold"

export PYTHONPATH=$BASE
export WANDB_API_KEY="491091873caf3d64c3b062c4fee4297d4a19b359"

# Print current datetime:
echo "START" $(date +"%Y-%m-%d %H:%M:%S")

# Print node list:
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NODELIST=$SLURM_NODELIST"

export MASTER_ADDR=$(echo $SLURM_NODELIST | cut -d ',' -f 1)
export MASTER_PORT=$SLURM_JOB_ID

# Set number of threads to use for parallel regions:
export OMP_NUM_THREADS=1

srun --export=ALL \
    torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --training_dirpath _runs/training_rundir \
    --pdb_mmcif_chains_filepath /data/pdb_mmcif/processed/chains.csv \
    --pdb_mmcif_dicts_dirpath /data/pdb_mmcif/processed/dicts \
    --pdb_obsolete_filepath /data/pdb_mmcif/processed/obsolete.dat \
    --pdb_alignments_dirpath /data/open_protein_set/processed/pdb_alignments \
    --seed 1234567890 \
    --num_train_iters 2000 \
    --val_every_iters 40 \
    --local_batch_size 1 \
    --base_lr 1e-3 \
    --warmup_lr_init 1e-5 \
    --warmup_lr_iters 0 \
    --num_train_dataloader_workers 14 \
    --num_val_dataloader_workers 2 \
    --disable_warmup \
    --distributed
# --initialize_parameters_from /data/mlperf_hpc_openfold_resumable_checkpoint.pt \
