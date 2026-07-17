#!/bin/bash

# --- SLURM RESOURCE REQUESTS ---
#SBATCH --job-name=MRMR_MLP_D2_SS_NEE_Modeling     # Name of your job
#SBATCH --partition=lncc-h100_shared     # The partition (queue) to run on
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks-per-node=1      # Request 1 task (your python script)
#SBATCH --cpus-per-task=24       # Request all CPUs (1* 24 = 24) for this one task
#SBATCH --gres=gpu:1             # Request 1 GPUs
#SBATCH --mem=200G                   # Request 128 GB of RAM
#SBATCH --time=24:00:00          # Time limit
#SBATCH --mail-type=END,FAIL      # Mail events (END, FAIL)
#SBATCH --mail-user=lucas.bauer@inpe.br # Replace with your email address
#SBATCH --output=mlp_d2_ss_mrmr_%j.log # Output file

#=======================================================================
# JOB COMMANDS
# This section tells Slurm what to do with the allocated resources.
#=======================================================================
echo "==============================================================="
echo "Job Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "==============================================================="


# 3. Run your Python script
echo "Starting Python script for NEE prediction using MLP..."
python -u mlp_qc75_ss_mrmr.py

echo "==============================================================="
echo "Job Finished at $(date)"
echo "==============================================================="