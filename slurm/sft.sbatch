#!/bin/bash
#SBATCH --job-name=formal_disco_sft
#SBATCH --account=kempner_fellows
#SBATCH --partition=kempner_h100_priority3
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "$FORMAL_DISCO_ROOT" || { echo "FORMAL_DISCO_ROOT not found"; exit 1; }

source venv/bin/activate
python distill.py sft
