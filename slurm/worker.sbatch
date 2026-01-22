#!/bin/bash
#SBATCH --job-name=disco_worker
#SBATCH --account=kempner_fellows
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "$FORMAL_DISCO_ROOT" || { echo "FORMAL_DISCO_ROOT not found"; exit 1; }

source venv/bin/activate

SERVER_INDEX=1
export VLLM_SERVER_LIST_FILE=$HOME/vllm_servers
export VLLM_BASE_URL=$(sed -n "${SERVER_INDEX}p" "$VLLM_SERVER_LIST_FILE")

echo "Using vLLM server $VLLM_BASE_URL"

# This looks up the agenda server address from a file in $HOME,
# and picks a random vllm server from $HOME/vllm_servers to attach to.
python scheduler.py +agenda=distributed +scheduler=readme_ideas_local
