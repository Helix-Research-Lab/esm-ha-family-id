#!/bin/bash
#SBATCH --job-name=ha_noise_sweep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# For a GPU run instead, swap the partition line for:
#   #SBATCH --partition=gpu
#   #SBATCH --gres=gpu:1

# Direct env python: `source activate` fails inside SLURM jobs.
PY=/oak/stanford/groups/rbaltman/rnarain7/envs/ha_research/bin/python

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Safe to resubmit: finished variants are skipped.
$PY noise_sweep.py --out results
