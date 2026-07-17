#!/bin/bash

# --- SLURM RESOURCE REQUESTS ---
#SBATCH --job-name=7K_MRMR_RF_D2_SS      # Name of your job
#SBATCH --partition=lncc-cpu_amd     # The partition (queue) to run on
#SBATCH --nodes=1                   # only one computer
#SBATCH --ntasks-per-node=1         # One task per computer
#SBATCH --cpus-per-task=48          # Request CPU cores
#SBATCH --mem=128G                   # Request 128 GB of RAM
#SBATCH --time=24:00:00           # Set a time limit of 1 day (D-HH:MM:SS)
#SBATCH --mail-type=END,FAIL      # Mail events (END, FAIL)
#SBATCH --mail-user=lucas.bauer@inpe.br # Replace with your email address
#SBATCH --output=mrmr_rf_d2_ss_7k%j.log # Output file

#=======================================================================
# JOB COMMANDS
# This section tells Slurm what to do with the allocated resources.
#=======================================================================
echo "==============================================================="
echo "Job Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "==============================================================="

# 1. Prepare the environment
#module purge      # Clean up any previously loaded modules
#module load anaconda3   

# 3. Run your Python script
echo "Starting Python script for NEE prediction..."
python -u rf_qc75_ss_mrmr_7k.py

echo "==============================================================="
echo "Job Finished at $(date)"
echo "==============================================================="