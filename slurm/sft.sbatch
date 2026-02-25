#!/bin/bash
#SBATCH --job-name=sft_formal_disco
#SBATCH --account=kempner_poesia_lab
#SBATCH --partition=kempner_h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "$FORMAL_DISCO_ROOT" || { echo "FORMAL_DISCO_ROOT not found"; exit 1; }

source venv/bin/activate
accelerate launch distill.py sft
