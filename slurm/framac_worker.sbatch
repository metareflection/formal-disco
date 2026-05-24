#!/bin/bash
#SBATCH --job-name=framac_worker
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "$FORMAL_DISCO_ROOT" || { echo "FORMAL_DISCO_ROOT not found"; exit 1; }

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
        echo "Run 'why3 config detect' from a node where $HOME/.opam/default/bin/$P binaries are reachable." >&2
        echo "---why3 config list-provers---" >&2
        echo "$PROVERS" >&2
        exit 1
    fi
done

source ../venv/bin/activate

SERVER_INDEX=$1
export VLLM_SERVER_LIST_FILE=$HOME/vllm_servers
export VLLM_BASE_URL=$(sed -n "${SERVER_INDEX}p" "$VLLM_SERVER_LIST_FILE")

echo "Using vLLM server $VLLM_BASE_URL"

python scheduler.py +agenda=distributed +scheduler=disco_framac_vllm
