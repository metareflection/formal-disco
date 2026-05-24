#!/bin/bash
#SBATCH --job-name=sn_framac
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=2-0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

N_WORKERS=32
SCHEDULE=disco_framac_vllm
RUN_NAME=sn-framac__opus-fewshot-sft

cd "$FORMAL_DISCO_ROOT" || { echo "FORMAL_DISCO_ROOT not found"; exit 1; }

# --- Frama-C / Why3 prover env -----------------------------------------------
# Why3, alt-ergo, cvc5 live in shared NFS via opam, so this works on any node.
if [ -d "$HOME/.opam/default/bin" ]; then
    export PATH="$HOME/.opam/default/bin:$PATH"
fi
if ! command -v why3 >/dev/null; then
    echo "FATAL: why3 not on PATH (expected $HOME/.opam/default/bin/why3)" >&2
    exit 1
fi
PROVERS=$(why3 config list-provers 2>/dev/null)
for P in "Alt-Ergo" "CVC5"; do
    if ! echo "$PROVERS" | grep -q "$P"; then
        echo "FATAL: prover '$P' not registered in why3.conf." >&2
        echo "Run 'why3 config detect' from a node where $HOME/.opam/default/bin/$P is reachable." >&2
        echo "---why3 config list-provers---" >&2
        echo "$PROVERS" >&2
        exit 1
    fi
done

# --- vLLM --------------------------------------------------------------------
source ../vllm-venv/bin/activate

PORT=30000
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_LOGGING_LEVEL=info

# MODEL=/n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/qwen2.5-32B_opus-distilled/merged
# MODEL=/n/holylabs/amin_lab/Lab/kempner-accelerator/models/qwen3coder-sft-ep3-v3/merged
# MODEL=/n/holylabs/kempner_poesia_lab/Lab/formal-disco/models/qwen3-coder-claude+self-v7/merged
# MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
MODEL=/n/holylabs/amin_lab/Lab/namin/sft-out-framac-opus-fewshot/merged

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

# 2. Wait for vllm to be ready
echo "Waiting for vllm server (PID $VLLM_PID) to be ready..."
while true; do
    sleep 30
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server is ready."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vllm process died unexpectedly. Exiting."
        exit 1
    fi
    echo "Not ready yet, retrying in 30s..."
done

# --- Disco runtime -----------------------------------------------------------
source ../venv/bin/activate

mkdir -p agendas

# 3. Run agenda server (Frama-C agenda).
AGENDA_ADDRESS_PATH=agenda_server_${RUN_NAME}.json
python agenda_server.py \
    agenda=local_framac \
    server.server_address_path=${AGENDA_ADDRESS_PATH} \
    agenda.checkpoint_path="agendas/agenda-${RUN_NAME}.pkl" \
    agenda.checkpoint_interval=1000 \
    agenda.max_attempts=100000 \
    &
AGENDA_PID=$!

sleep 10

# 4. Run workers in parallel (each in the background).
WORKER_PIDS=()
for ((i=1; i<=N_WORKERS; i++)); do
    python scheduler.py \
        +agenda=distributed \
        +scheduler="${SCHEDULE}" \
        agenda.server_address_path="${AGENDA_ADDRESS_PATH}" \
        &
    WORKER_PIDS+=($!)
done

# 5. Wait for all workers to finish (they exit when the agenda hits max_attempts).
echo "Waiting for ${#WORKER_PIDS[@]} worker(s) to finish..."
wait "${WORKER_PIDS[@]}"
echo "All workers done."

# 6. Tear down agenda server.
echo "Shutting down agenda..."
kill $AGENDA_PID
wait $AGENDA_PID

echo "Done."
