[![arXiv](https://img.shields.io/badge/arXiv-2607.04631-pink.svg)](https://arxiv.org/abs/2607.04631)

# Formal Disco

**Scalable open-ended generation of formally verified programs**

Formal Disco is a distributed system where LLM-based workers collaborate through a shared *agenda* (a task queue + object database) to generate, repair, and extend formally verified programs, using the verifier as a scalable quality signal. Guided by a *principle of maximum entropy* over program features, the system self-improves via iterative fine-tuning to produce programs that are both increasingly likely to verify and increasingly diverse. We instantiate it for three verification-aware languages - **Dafny**, **Verus** (Rust), and **Frama-C** - and use it to generate large synthetic datasets for each language.

This repository accompanies the paper *Formal Disco: Scalable Open-Ended Generation of Formally Verified Programs*. The generated datasets are available on HuggingFace: <https://huggingface.co/collections/metareflection/formal-disco>.

## Overview

One *run* of the system starts an agenda and schedules three classes of workers (Section 3.2 of the paper), each backed by an LLM:

- **Initiator** (`worker/initiator.py`): samples a *seed context* — a random GitHub README plus up to 2 documentation snippets from the language's reference (`language/*/features/*.txt`) — and prompts the LLM to write a small initial verified program. Depending on the verifier's outcome, it enqueues an `extend` or `repair` task.
- **Fixer** (`worker/llm_fixer.py`): claims `repair` tasks, reads the failing program and verifier feedback, and produces a diff to fix the errors (up to 3 attempts per task).
- **Extender** (`worker/editor_worker.py`): claims `extend` tasks and produces a diff that grows a working program in complexity (new lemmas, methods, features).

Every LLM call is recorded in the agenda as a *distillation example* (prompt, response, outcome), which powers both distillation from frontier models and self-improvement of open models via entropy-maximizing SFT (Section 3.3).

## Setup

```bash
pip install -r requirements.txt
```

You also need the verifier(s) for the language(s) you want to run on your `PATH`:

- **Dafny**: [`dafny`](https://github.com/dafny-lang/dafny) (we invoke `dafny verify`)
- **Verus**: [`verus`](https://github.com/verus-lang/verus)
- **Frama-C**: [`frama-c`](https://frama-c.com/) with the WP plugin (and provers via `why3`)

Two data dependencies:

- **GitHub READMEs**: the Initiator samples from `data/gh-readmes-100k.jsonl`, a dataset of 100k README files we sampled from Google BigQuery's GitHub dataset (one JSON object per line with the repository name and README content). We will upload this file soon.
- **W&B (optional)**: runs log metrics to [Weights & Biases](https://wandb.ai) by default; pass `agenda/logger=noop` when running the agenda server if you do not want to log to wandb.

If you do use wandb, you have to be logged in (`wandb login`).

All configuration uses [Hydra](https://hydra.cc/): see `config/`. LLM providers are configured under `config/llm/` (OpenAI-compatible, AWS Bedrock, Vertex, vLLM, Ollama) via LangChain, so switching providers is easy. For the paper, we ran Claude models through AWS Bedrock, and inference on fine-tuned Qwen using vLLM serving the local model through its OpenAI-compatible API.

## Running the discovery system

A quick test with some dummy (no-LLM) workers:

```bash
timeout 5 python scheduler.py +agenda=local +scheduler=example
```

A real local run with the three workers on Dafny, using Claude Opus via Bedrock:

```bash
python scheduler.py +agenda=local +scheduler=disco_opus
```

This runs until the agenda's `max_attempts` task attempts are reached, checkpointing periodically to a pickle file (see `checkpoint_path` in `config/agenda/local.yaml`). Other languages and models:

```bash
# Verus with Claude Opus
python scheduler.py +agenda=local_verus +scheduler=disco_opus scheduler.language=verus

# Frama-C with a local fine-tuned model served by vLLM (VLLM_BASE_URL env var)
python scheduler.py +agenda=local_framac +scheduler=disco_framac_vllm

# Monolithic single-agent baseline (ablation in §4.4)
python scheduler.py +agenda=local +scheduler=monolithic_claude_opus
```

At any point, dump the generated artifacts (programs, ideas, all historical versions) from a checkpoint to disk:

```bash
python materialize.py materialize agenda.pkl discoveries/
```

### Distributed mode

For large runs, a single agenda server coordinates many worker processes (on one node or across a cluster). A complete run with a local model has three pieces:

```bash
# 1. (If using a local model) Serve it with vLLM. The vllm configs expect the
#    served model name "llm" and read the endpoint from VLLM_BASE_URL.
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
    --max-model-len 8K --served-model-name llm --port 30000
export VLLM_BASE_URL=http://localhost:30000/v1

# 2. Start the agenda server. It writes its address to a JSON file that
#    workers read to discover it.
python agenda_server.py \
    server.server_address_path=agenda_server_address.json \
    agenda.checkpoint_path=agendas/my-run.pkl \
    agenda.max_attempts=100000 &

# 3. Start as many workers as you like, on any machine that can reach the server.
for i in $(seq 1 32); do
    python scheduler.py +agenda=distributed +scheduler=disco_vllm \
        agenda.server_address_path=agenda_server_address.json &
done
wait
```

Workers exit once the agenda reaches `max_attempts` (the agenda raises an exception on every request when that happens, and workers know to shut down). On a cluster, wrap each of the three pieces in your job scheduler of choice (e.g., slurm); they only need a shared filesystem (or a fixed host/port) for the address file.

## Distillation and self-improvement

Inspect the distillation examples collected in a run:

```bash
python distill.py stats -d agenda.pkl
```

Fine-tune an open model (default: Qwen2.5-Coder-32B-Instruct with LoRA) on collected examples:

```bash
python distill.py sft data=dafny-it0 language=dafny output_dir=sft-out
```

The `config/data/*-it*.yaml` configs list the agenda checkpoints that feed each self-improvement iteration (Dafny, Verus, and Frama-C series). Setting `top_surprisal_fraction: 0.33` (the default) enables the entropy-maximization selection from §3.3: only the top third of successful examples, ranked by minimum surprisal rank of the programs they generated, are used for training. The self-improvement loop of the paper alternates `distill.py sft` (on all data so far) with a new discovery run (`+scheduler=disco_vllm`, serving the fine-tuned model).

## Downstream tasks

Task definitions live in `tasks/` - `fixer.py` (the logical annotation task in Dafny: strip assertions/invariants/etc, predict them as a diff), `lemma_synth.py` (hollow a lemma body, generate the proof; language-agnostic), `verus_fixer.py` (logical annotation task for Verus). The pipeline is:

```bash
# 1. Extract training examples from generated programs into train/val pickles
python extract.py task=fixer data=agenda pickle=agenda.pkl output_prefix=fixer

# 2. Fine-tune on the extracted examples -- note that dafny_fixer is a file you should have in config/data/dafny_fixer.yaml
python distill.py sft data=dafny_fixer output_dir=sft-fixer

# 3. Evaluate on existing benchmarks (DafnyBench / VerusBench)
python eval.py task=fixer data=dafnybench llm=vllm
```

Baselines against Claude/OpenAI models work by switching `llm=`; `baselines/` additionally contains the scaffolding-free base-LLM annotator and converters for the SAFE/VeruSyn external datasets.

## Analysis and paper figures

`analysis.py` computes all corpus statistics (success rates, feature entropies, complexity metrics, rarefaction curves, Pass@k) from agenda checkpoints and eval result JSONs, and renders plots with Vega-Lite/Altair. Figure regeneration is orchestrated by a small make-style runner we have under the `make` directory (this will be made into its own independent library soon, since it's generally useful like `make`):

```bash
python -m make --list                        # show all targets
python -m make rarefaction-comparison        # Fig. 4 + appendix rarefaction figures
python -m make fixer-pass-at-k lemma-pass-at-k   # Figs. 5-6, from results/final/*.json
python -m make ablation-gh-rarefaction ablation-docs-rarefaction  # Fig. 7
python analysis.py plot-iterations           # Figs. 2-3, from final-runs/agendas/
python scripts/ablation_stats.py             # monolithic-vs-workers ablation numbers (4.4)
```

These read the run checkpoints (`final-runs/agendas/`, `agendas/for-ablations/`) and evaluation results (`results/final/`), which we will share soon (the program datasets derived from the agenda files are in the HuggingFace collection).

`web/` contains an optional Vite/React viewer for inspecting eval results, examples, and some diversity reports (drag-and-drop the JSON files produced by `analysis.py` and `eval.py`): `cd web && npm install && npm run dev`.

## Tests

```bash
pytest tests/
```

Backend tests (`test_dafny_backend.py`, `test_verus_backend.py`, `test_framac_backend.py`) require the corresponding verifier on your `PATH`.

## Repository layout

```
agenda.py, agenda_distributed.py   # task queue + object DB (local and RPC-distributed)
agenda_server.py                   # distributed agenda server entry point
scheduler.py                       # round-robin worker scheduler (main entry point)
worker/                            # Initiator, Fixer (LLMFixer), Extender (EditorWorker), baselines
language/{dafny,verus,framac}/     # per-language backend: verification, features, prompts, doc snippets
patch.py                           # LLM-facing text diff format + object version patches
distill.py                         # distillation stats + SFT with entropy-maximizing selection
tasks/ extract.py eval.py          # downstream task definitions, data extraction, evaluation
analysis.py, make/                 # corpus analysis and paper figure generation
materialize.py                     # dump agenda objects (all versions) to disk
config/                            # Hydra configs (scheduler, agenda, llm, data, task)
tests/                             # pytest suite
```

## License

MIT

## Citation

```
@misc{poesia2026formaldiscoscalableopenended,
      title={Formal Disco: Scalable Open-Ended Generation of Formally Verified Programs}, 
      author={Gabriel Poesia and Simon Henniger and Tzu-Han Hsu and Yilun Du and Nada Amin},
      year={2026},
      eprint={2607.04631},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.04631}, 
}
```
