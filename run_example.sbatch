#!/bin/bash
# Example SLURM submission script (generic template).
# Replace <ACCOUNT> and <PARTITION> with values for your cluster,
# and point PYTHON at your environment's interpreter.
#SBATCH --job-name=specmorph
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

PYTHON=python   # e.g. path to a conda/venv python with PyTorch installed

# Example: run a single-domain continual-learning experiment.
$PYTHON src/experiments/realgraph_pmnist.py
