#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --account=kempner_fellows
#SBATCH --partition=kempner_h100_priority3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "$FORMAL_DISCO_ROOT" || { echo "FORMAL_DISCO_ROOT not found"; exit 1; }

source vllm-venv/bin/activate

# Runs a simple wandb monitor to collect system metrics
python monitor.py &

# Write host and port to an NFS file, so clients can find servers.
HOSTNAME=$(hostname)
PORT=30000
echo "http://${HOSTNAME}:${PORT}/v1" >> "$HOME/vllm_servers"

#vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
# vllm serve /n/holylabs/kempner_fellows/Lab/poesia/accelerator-share/models/qwen3coder-sft-v1/ \
# vllm serve /n/holylabs/LABS/amin_lab/Lab/namin/sft-out-claude-free-2/merged/ \ # /n/holylabs/kempner_fellows/Lab/poesia/accelerator-share/models/qwen3coder-sft-v1/ \
# vllm serve /n/holylabs/amin_lab/Lab/kempner-accelerator/models/qwen3coder-sft-ep3-v3/merged \
# vllm serve openai/gpt-oss-20b \
# vllm serve openai/gpt-oss-20b \
# vllm serve /n/holylabs/amin_lab/Lab/kempner-accelerator/models/deepseekcoder-v2-lite-sft-v1/merged \
# vllm serve /n/holylabs/amin_lab/Lab/kempner-accelerator/models/gpt-oss-20b-sft-v1/merged \
# vllm serve /n/holylabs/amin_lab/Lab/kempner-accelerator/models/gpt-oss-20b-distill-v2/merged \
# vllm serve /n/holylabs/amin_lab/Lab/kempner-accelerator/models/qwen3-coder-ideafix-v6/merged \
vllm serve /n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/qwen3-coder-claude+self-v7/merged \
    --max-model-len 8K \
    --tensor-parallel-size 4 \
    --dtype auto \
    --served-model-name llm \
    --host 0.0.0.0 \
    --port $PORT
