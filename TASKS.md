# Task System

## Overview

The discovery system (`scheduler.py`) produces agenda checkpoints (e.g. `agenda-run3.pkl`) containing verified Dafny programs. The task system turns these into training data, trains models, and evaluates them. The full flow:

```
                        agenda checkpoint (agenda-run3.pkl)
                                    |
                        +-----------+-----------+
                        |                       |
                  extract.py               extract.py
                  task=fixer            task=lemma_synth
                        |                       |
                fixer_train.pkl          lemma_train.pkl
                fixer_val.pkl            lemma_val.pkl
                        |                       |
                        +-----------+-----------+
                                    |
                            distill.py sft
                      data=[fixer_train.pkl, ...]
                                    |
                          model checkpoint
                           (served via vLLM)
                                    |
                        +-----------+-----------+
                        |           |           |
                    eval.py     eval.py     eval.py
                  task=fixer  task=implement task=lemma_synth
                   llm=vllm    llm=vllm      llm=vllm
                        |           |           |
                     success     success     success
                      rates       rates       rates
```

### Step by step

**1. Extract training/eval data from an agenda checkpoint.**

Each task knows how to generate its own examples from verified programs:
- **fixer**: strips hints (invariants, assertions) from verified programs to create broken/fixed pairs
- **lemma_synth**: hollows out lemma bodies to create synthesis challenges
- **implement**: pairs ideas with their verified implementations

```bash
python extract.py task=fixer output_prefix=fixer
python extract.py task=lemma_synth output_prefix=lemma
```

The default data source for extraction is `data=agenda` (see `config/data/agenda.yaml`). To use a different agenda checkpoint:

```bash
python extract.py task=fixer 'data.sources=[{type: pickle, path: agenda-run3.pkl, extract_from_verified: true}]' output_prefix=fixer
```

This produces `*_train.pkl` and `*_val.pkl` files, split by program (so no data leakage between train and eval).

**2. Train a model on the extracted data.**

```bash
python distill.py sft 'data=[fixer_train.pkl, lemma_train.pkl]'
```

This trains an SFT model (LoRA on Qwen by default) using each task's `to_training_record()` to build the chat-format training examples.

**3. Serve the trained model and evaluate it.**

```bash
# Serve the model (outside this repo, e.g. with vLLM)
# Then evaluate on held-out val splits:
python eval.py task=fixer llm=vllm data=fixer_val       # fixer on val split
python eval.py task=lemma_synth llm=vllm data=lemma_val  # lemma on val split
python eval.py task=implement llm=vllm data=implement_val # implement on val split

# Or evaluate on external benchmarks:
python eval.py task=fixer llm=vllm                       # fixer on DafnyBench (default)
```

Each eval reports a success rate. Compare across model checkpoints to pick the best one.

## Architecture

Every task subclasses `EvaluationTask` from `tasks/__init__.py` and implements three methods:

```python
class EvaluationTask(ABC):
    name: str           # e.g. "fixer"
    prompt_type: str    # e.g. "repair"

    def extract_examples(self, sources: list[dict]) -> list[dict]
    def evaluate_one(self, llm, example) -> dict
    def to_training_record(self, example) -> dict | None
```

Sources are passed into `extract_examples` from the `data` config group (`config/data/*.yaml`), not stored on the task object.

| Method | Purpose |
|--------|---------|
| `extract_examples` | Load/generate examples from the given data sources |
| `evaluate_one` | Run the model on one example, return `{"success": bool, ...}` |
| `to_training_record` | Convert a distill example into `{"messages": [...], "completion": str}` for SFT |

The base class also provides:

- **`evaluate(llm, examples)`** — runs `evaluate_one` in a loop with progress bar and optional wandb logging.
- **`evaluate_with_retries(llm, example, max_attempts)`** — shared generate-verify-retry loop for iterative tasks (fixer, lemma_synth). Subclasses override `generate`, `apply_response`, `verify`, and `update_for_retry` hooks.
- **`discover_tasks()`** — auto-discovers all task classes in `tasks/*.py`.

## Available Tasks

### Fixer (`tasks/fixer.py`)

Repairs broken Dafny programs by iteratively generating and applying diffs.

**Data sources:**
- `dfy` glob — load `.dfy` files that fail verification, e.g. from DafnyBench
- `pickle` with `prompt_types: [repair]` — pre-made repair examples
- `pickle` with `extract_from_verified: true` — strip hints from verified programs to create broken/fixed pairs

**Evaluation:** Iterative (up to `max_attempts`). Each attempt: prompt LLM for a diff, apply it with `apply_text_diff`, verify with Dafny.

**Config:**
```yaml
# config/task/fixer.yaml
_target_: tasks.fixer.FixerTask
max_attempts: 3
filter_trivial: true
```

**Usage:**
```bash
python eval.py task=fixer llm=openai                  # DafnyBench (default data=dafnybench)
python eval.py task=fixer llm=vllm data=fixer_val     # In-distribution val split
```

### Implement (`tasks/implement.py`)

Generates Dafny programs from idea/specification descriptions.

**Data sources:**
- `pickle` with `prompt_types: [implement]` — idea-to-code examples
- `dfy` glob — extract `// Idea prompt:` comments from `.dfy` files

**Evaluation:** Single-shot. Prompt LLM with idea, verify generated code.

**Config:**
```yaml
# config/task/implement.yaml
_target_: tasks.implement.ImplementTask
```

**Usage:**
```bash
python eval.py task=implement llm=vllm data=implement_val
```

### Lemma Synthesis (`tasks/lemma_synth.py`)

Fills in empty lemma bodies so the program verifies.

**Data sources:**
- `pickle` with `prompt_types: [lemma_synth]` — pre-made lemma examples
- `pickle` with `extract_from_verified: true` — hollow lemma bodies from verified programs

**Evaluation:** Iterative (up to `max_attempts`). Each attempt: prompt LLM for body, insert into program, verify.

**Config:**
```yaml
# config/task/lemma_synth.yaml
_target_: tasks.lemma_synth.LemmaSynthTask
max_attempts: 3
```

**Usage:**
```bash
python eval.py task=lemma_synth llm=openai data=lemma_val num_examples=50
```

## How to Run

There are three entry points: `eval.py` (evaluate a model), `extract.py` (generate training data), and `distill.py sft` (train a model). All use Hydra for configuration.

### eval.py — Evaluate a model on a task

`eval.py` composes three Hydra config groups: `task` (from `config/task/`), `llm` (from `config/llm/`), and `data` (from `config/data/`).

```bash
# Evaluate fixer on DafnyBench (default data=dafnybench)
python eval.py task=fixer llm=openai

# Evaluate fixer on val split
python eval.py task=fixer llm=vllm data=fixer_val

# Evaluate lemma synthesis on val split, limit to 50 examples
python eval.py task=lemma_synth llm=openai data=lemma_val num_examples=50

# Evaluate implement task
python eval.py task=implement llm=vllm data=implement_val

# Evaluate on an arbitrary dfy glob
python eval.py task=fixer llm=vllm data=glob glob="../dafny-vfp/autogen/bench*minimized/**/*.dfy"

# Save results to JSON, disable wandb
python eval.py task=fixer llm=openai output=results.json wandb=false

# Override LLM model
python eval.py task=fixer llm=openai llm.code.model=gpt-4o
```

Available tasks: `fixer`, `fixer_indist`, `implement`, `lemma_synth` (defined in `config/task/`).
Available LLMs: `openai`, `aws`, `awsbest`, `vllm`, `ollama` (defined in `config/llm/`).
Available data presets: `dafnybench`, `fixer_val`, `lemma_val`, `implement_val`, `agenda` (defined in `config/data/`).

Common options:

| Override | Default | Description |
|----------|---------|-------------|
| `data=name` | dafnybench | Data source preset |
| `num_examples=N` | all | Limit to first N examples |
| `seed=N` | 42 | Random seed for shuffling examples |
| `output=path.json` | none | Save results to JSON file |
| `verbose=true` | false | Verbose logging |
| `wandb=false` | true | Disable wandb logging |
| `task.max_attempts=N` | 3 | Max repair attempts (fixer, lemma_synth) |

### extract.py — Generate training data

`extract.py` uses the `task` and `data` config groups (no LLM needed). The default data source is `data=agenda`.

```bash
# Extract fixer training data from the default agenda checkpoint
python extract.py task=fixer output_prefix=fixer

# Extract lemma examples
python extract.py task=lemma_synth output_prefix=lemma

# Extract from a specific agenda pickle
python extract.py task=fixer \
  'data.sources=[{type: pickle, path: agenda-run3.pkl, extract_from_verified: true}]' \
  output_prefix=fixer

# Extract implement examples from .dfy files
python extract.py task=implement data=glob glob="autogen/**/*.dfy" output_prefix=implement
```

Produces `{output_prefix}_train.pkl` and `{output_prefix}_val.pkl`, split by program path.

| Override | Default | Description |
|----------|---------|-------------|
| `data=name` | agenda | Data source preset |
| `output_prefix=name` | task name | Prefix for output pickle files |
| `val_fraction=0.2` | 0.2 | Fraction of programs held out for validation |
| `seed=N` | 42 | Random seed for the split |

### distill.py sft — Train a model

Uses `config/distill.yaml` for training hyperparameters.

```bash
# Train on agenda checkpoint pickles (as configured in config/distill.yaml)
python distill.py sft

# Train on specific pickle files
python distill.py sft data=fixer_train.pkl

# Train on multiple pickle files
python distill.py sft 'data=[fixer_train.pkl, lemma_train.pkl]'

# Override training parameters
python distill.py sft data=fixer_train.pkl learning_rate=1e-4 num_train_epochs=3
```

## Data Sources

Data sources are configured via the `data` Hydra config group (`config/data/*.yaml`). Each preset contains a `sources` list. Available presets:

| Preset | Description |
|--------|-------------|
| `dafnybench` | DafnyBench `.dfy` files (default for eval) |
| `fixer_val` | Fixer val split pickle |
| `lemma_val` | Lemma synth val split pickle |
| `implement_val` | Implement val split pickle |
| `agenda` | Agenda checkpoint for extraction (default for extract) |
| `glob` | Arbitrary `.dfy` glob — set `glob=` on the command line (see below) |

The `glob` preset uses Hydra interpolation (`${glob}`) to read from a top-level config key, so you can point at any `.dfy` files without quoting nested YAML:

```bash
python eval.py task=fixer llm=vllm data=glob glob="../dafny-vfp/autogen/**/*.dfy"
python extract.py task=implement data=glob glob="autogen/**/*.dfy" output_prefix=implement
```

To add a new preset, create `config/data/my_data.yaml`:

```yaml
sources:
  - type: pickle
    path: my_data.pkl
    prompt_types: [repair]
```

Each source has a `type` field:

### Pickle sources (`type: pickle`)

Load examples from agenda checkpoint pickle files.

| Key | Type | Description |
|-----|------|-------------|
| `path` | str | Path to pickle file |
| `prompt_types` | list[str] | Filter by prompt type (e.g. `[repair]`) |
| `success_only` | bool | Only include successful examples |
| `extract_from_verified` | bool | Generate examples from verified programs (task-specific) |

### Dfy glob sources (`type: dfy`)

Load `.dfy` files matching a glob pattern.

| Key | Type | Description |
|-----|------|-------------|
| `glob` | str | Glob pattern (e.g. `"DafnyBench/**/*.dfy"`) |

## Adding a New Task

1. Create `tasks/my_task.py`:

```python
from tasks import EvaluationTask

class MyTask(EvaluationTask):
    name = "my_task"
    prompt_type = "my_prompt_type"

    def extract_examples(self, sources: list[dict]):
        # Load and return examples as list[dict]
        # Sources are passed in from the data config group
        ...

    def evaluate_one(self, llm, example):
        # Run model on one example
        # Return {"success": bool, ...}
        ...

    def to_training_record(self, example):
        # Return {"messages": [...], "completion": str}
        ...
```

2. Create `config/task/my_task.yaml`:

```yaml
_target_: tasks.my_task.MyTask
```

3. Create a data preset `config/data/my_data.yaml`:

```yaml
sources:
  - type: pickle
    path: my_data.pkl
    prompt_types: [my_prompt_type]
```

4. Run it:

```bash
python eval.py task=my_task data=my_data llm=openai
```

The task is automatically discovered by `discover_tasks()` and usable from `eval.py`, `extract.py`, and `distill.py`.

## Shared Utilities

Tasks build on these shared modules:

- **`prompt.py`** — prompt strings for all task types (`system_repair`, `format_implement_user`, etc.)
- **`patch.py`** — `apply_text_diff()` for diff application, `TEXT_DIFF_EXAMPLE` for few-shot examples
- **`dafny.py`** — `DafnyProgram` for verification
- **`code_output_parser.py`** — `CodeOutputParser` for extracting code from LLM responses
- **`eval_common.py`** — LLM creation, argparse helpers, wandb helpers, pickle loading
- **`distill_common.py`** — pickle I/O, hint removal, text diff computation, Dafny error capture
