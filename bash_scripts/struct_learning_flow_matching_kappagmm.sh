#!/bin/bash
#SBATCH --job-name=d$1kappa$2         # Job name
#SBATCH --output=slurm_logs/output_%j.log     # Output file (%j = job ID)
#SBATCH --error=slurm_logs/error_%j.log       # Error file
## SBATCH --time=01:00:00            # Time limit (hh:mm:ss)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G                   # Memory per node
#SBATCH --gres=gpu:1 # no gpus here

# Load modules if needed
# module load python/3.9

# Print job info
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"


# do installs
# pip install torchsde
# pip install -e weight-learning

# generate models
#python train_log.py --dim=${1} --batch_size=512 --max_epoch=20000 --nb_save=50

# generate trajectories

uv run python scripts/struct_learning_flow_matching_kappagmm.py \
--dim=$1 \
--kappa=$2 \
--iterations=2000 \
--nb_log_points=100 \
--interval=100 \
--log_scale=log \
--nsamples=10000 \
--bwd_repeat=10 


echo job ${1}
echo "Job finished at $(date)"

