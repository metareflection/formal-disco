#!/bin/bash
#SBATCH --job-name=agenda_server
#SBATCH --account=kempner_fellows
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "$FORMAL_DISCO_ROOT" || { echo "FORMAL_DISCO_ROOT not found"; exit 1; }

source venv/bin/activate

# This starts the server and writes its address to a file in $HOME, so workers can find it.
python agenda_server.py
