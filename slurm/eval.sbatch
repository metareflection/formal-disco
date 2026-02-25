#!/bin/bash
#SBATCH --job-name=eval_formal_disco
#SBATCH --account=kempner_poesia_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "$FORMAL_DISCO_ROOT" || { echo "FORMAL_DISCO_ROOT not found"; exit 1; }

source vllm-venv/bin/activate

PORT=30000
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_LOGGING_LEVEL=debug

# MODEL=/n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/qwen3-coder-nontrivial-fixer+lemma-v1/merged
MODEL_NAME=qwen3-coder-nontrivial-fixer-v1
MODEL=/n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/${MODEL_NAME}/merged
# MODEL=/n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/qwen3-coder-nontrivial-fixer-v2/merged
# MODEL=Qwen/Qwen2.5-Coder-14B-Instruct
# MODEL=/n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/qwen3-coder-nontrivial-fixer-v3/merged

# 1. Start vllm in the background
vllm serve $MODEL \
    --max-model-len 8K \
    --tensor-parallel-size 2 \
    --dtype auto \
    --served-model-name llm \
    --host 0.0.0.0 \
    --port $PORT &
VLLM_PID=$!

deactivate

# 2. Wait for the server to be ready
echo "Waiting for vllm server (PID $VLLM_PID) to be ready..."
while true; do
    sleep 30
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server is ready."
        break
    fi
    # Bail out if vllm died
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vllm process died unexpectedly. Exiting."
        exit 1
    fi
    echo "Not ready yet, retrying in 30s..."
done

source venv/bin/activate

# 3. Run evals in parallel (each in the background)
python eval.py task=fixer llm=vllm output=results/fixer-eval-${MODEL_NAME}.json &
EVAL_PIDS=($!)

# Example of a second eval running at the same time:
# python tasks/fixer/eval.py \
#     --llm-config vllm \
#     --output results/eval2.json \
#     --skip 100 \
#     &
# EVAL_PIDS+=($!)

# 4. Wait for all evals to finish
echo "Waiting for ${#EVAL_PIDS[@]} eval(s) to finish..."
wait "${EVAL_PIDS[@]}"
echo "All evals done."

# 5. Kill vllm server
echo "Shutting down vllm server..."
kill $VLLM_PID
wait $VLLM_PID
echo "Done."
