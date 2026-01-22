#!/bin/bash
#SBATCH --job-name=distill_claude
#SBATCH --account=kempner_fellows
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "$FORMAL_DISCO_ROOT" || { echo "FORMAL_DISCO_ROOT not found"; exit 1; }

source venv/bin/activate

python scheduler.py +agenda=local +scheduler=readme_ideas_aws agenda.checkpoint_path="agenda-claude-opus-distill.pkl"
