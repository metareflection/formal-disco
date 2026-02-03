# Task System

The `tasks/` package provides a unified framework for evaluating models on Dafny-related tasks, extracting training data, and generating SFT records. Each task is a self-contained module that handles its own data loading, evaluation logic, and training record formatting.

## Architecture

Every task subclasses `EvaluationTask` from `tasks/__init__.py` and implements three methods:

```python
class EvaluationTask(ABC):
    name: str           # e.g. "fixer"
    prompt_type: str    # e.g. "repair"

    def extract_examples(self, sources) -> list[dict]
    def evaluate_one(self, llm, example) -> dict
    def to_training_record(self, example) -> dict | None
```

| Method | Purpose |
|--------|---------|
| `extract_examples` | Load/generate examples from pickle files or `.dfy` file globs |
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

**Config examples:**
```yaml
# DafnyBench evaluation
_target_: tasks.fixer.FixerTask
max_attempts: 3
sources:
  - type: dfy
    glob: "DafnyBench/hints_removed/**/*.dfy"

# In-distribution evaluation
_target_: tasks.fixer.FixerTask
max_attempts: 3
sources:
  - type: pickle
    path: fixer_val.pkl
    prompt_types: [repair]
```

### Implement (`tasks/implement.py`)

Generates Dafny programs from idea/specification descriptions.

**Data sources:**
- `pickle` with `prompt_types: [implement]` — idea-to-code examples
- `dfy` glob — extract `// Idea prompt:` comments from `.dfy` files

**Evaluation:** Single-shot. Prompt LLM with idea, verify generated code.

**Config:**
```yaml
_target_: tasks.implement.ImplementTask
sources:
  - type: pickle
    path: distil-verified.pkl
    prompt_types: [implement]
```

### Lemma Synthesis (`tasks/lemma_synth.py`)

Fills in empty lemma bodies so the program verifies.

**Data sources:**
- `pickle` with `prompt_types: [lemma_synth]` — pre-made lemma examples
- `pickle` with `extract_from_verified: true` — hollow lemma bodies from verified programs

**Evaluation:** Iterative (up to `max_attempts`). Each attempt: prompt LLM for body, insert into program, verify.

**Config:**
```yaml
_target_: tasks.lemma_synth.LemmaSynthTask
max_attempts: 3
sources:
  - type: pickle
    path: lemma_val.pkl
    prompt_types: [lemma_synth]
```

## How to Run

There are three entry points: `eval.py` (evaluate a model), `extract.py` (generate training data), and `distill.py sft` (train a model). All use Hydra for configuration.

### eval.py — Evaluate a model on a task

`eval.py` composes two Hydra config groups: `task` (from `config/task/`) and `llm` (from `config/llm/`).

```bash
# Evaluate fixer on DafnyBench with OpenAI
python eval.py task=fixer llm=openai

# Evaluate with vLLM-served model
python eval.py task=fixer llm=vllm

# Evaluate lemma synthesis, limit to 50 examples
python eval.py task=lemma_synth llm=openai num_examples=50

# Evaluate implement task
python eval.py task=implement llm=vllm

# Evaluate fixer on in-distribution pickle data
python eval.py task=fixer_indist llm=openai

# Override the data source on the command line
python eval.py task=fixer llm=vllm \
  'task.sources=[{type: dfy, glob: "../dafny-vfp/autogen/bench*minimized/**/*.dfy"}]'

# Save results to JSON, disable wandb
python eval.py task=fixer llm=openai output=results.json wandb=false

# Override LLM model
python eval.py task=fixer llm=openai llm.code.model=gpt-4o
```

Available tasks: `fixer`, `fixer_indist`, `implement`, `lemma_synth` (defined in `config/task/`).
Available LLMs: `openai`, `aws`, `awsbest`, `vllm`, `ollama` (defined in `config/llm/`).

Common options:

| Override | Default | Description |
|----------|---------|-------------|
| `num_examples=N` | all | Limit to first N examples |
| `seed=N` | 42 | Random seed for shuffling examples |
| `output=path.json` | none | Save results to JSON file |
| `verbose=true` | false | Verbose logging |
| `wandb=false` | true | Disable wandb logging |
| `task.max_attempts=N` | 3 | Max repair attempts (fixer, lemma_synth) |

### extract.py — Generate training data

`extract.py` uses the task config group only (no LLM needed).

```bash
# Extract fixer training data from an agenda checkpoint
python extract.py task=fixer \
  'task.sources=[{type: pickle, path: agenda.pkl, extract_from_verified: true}]' \
  output_prefix=fixer

# Extract lemma examples from an agenda checkpoint
python extract.py task=lemma_synth \
  'task.sources=[{type: pickle, path: agenda.pkl, extract_from_verified: true}]' \
  output_prefix=lemma

# Extract implement examples from .dfy files
python extract.py task=implement \
  'task.sources=[{type: dfy, glob: "autogen/**/*.dfy"}]' \
  output_prefix=implement
```

Produces `{output_prefix}_train.pkl` and `{output_prefix}_val.pkl`, split by program path.

| Override | Default | Description |
|----------|---------|-------------|
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

Tasks accept a list of sources, each with a `type` field:

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

    def __init__(self, sources=None, **kwargs):
        self.sources = sources or []

    def extract_examples(self, sources=None):
        # Load and return examples as list[dict]
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
sources:
  - type: pickle
    path: my_data.pkl
    prompt_types: [my_prompt_type]
```

3. Run it:

```bash
python eval.py task=my_task llm=openai
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
