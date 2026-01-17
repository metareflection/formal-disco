# eval_implement.py

Evaluation script for the implement task on Claude-generated data.

This script evaluates a model's ability to implement Dafny programs from natural language ideas, using the same distribution as the Claude training data.

## Overview

The implement task works by:
1. Taking an idea/specification for a Dafny program (generated from GitHub repos)
2. Prompting an LLM to implement the idea as Dafny code
3. Running Dafny verification on the generated code
4. Comparing the outcome to what Claude achieved (ground truth)

This is the **honest eval** for models fine-tuned on Claude-generated data, as it tests on the same task distribution.

## Why This Eval?

The Claude training data contains three task types:
- `implement`: idea → full Dafny code
- `repair`: broken code → diff to fix it
- `extend`: working code → extended code

If you train on Claude data and eval on DafnyBench (which is a repair task with a specific diff format), you're testing on a **different distribution**. This eval tests on the same distribution as training.

## Data Source

The eval uses implement examples from Claude agenda checkpoint pickles:

| Pickle | Total Implement | Successful |
|--------|-----------------|------------|
| `claude/local-agenda-claude.pkl` | 1,601 | 33 (2.1%) |
| `claude/local-agenda-claude-opus.pkl` | 2,124 | 353 (16.6%) |

Examples include ideas like:
- "A verified calculator service with overflow protection"
- "HTTP status code classifier and validator"
- "Line splitter for character sequences"

## Usage

### Basic Usage

```bash
# Eval on successful implement examples only
python eval_implement.py \
    --llm-config vllm \
    --pickle claude/local-agenda-claude-opus.pkl \
    --success-only \
    --num-examples 50

# Eval on ALL implement examples (including failures)
python eval_implement.py \
    --llm-config vllm \
    --pickle claude/local-agenda-claude-opus.pkl \
    --num-examples 100

# Multiple pickle files
python eval_implement.py \
    --llm-config vllm \
    --pickle claude/local-agenda-claude.pkl \
    --pickle claude/local-agenda-claude-opus.pkl \
    --success-only
```

### LLM Configuration

**Option 1: Hydra config (`--llm-config`)**

Uses configs from `config/llm/`:
```bash
python eval_implement.py --llm-config vllm --pickle claude/local-agenda-claude-opus.pkl
python eval_implement.py --llm-config openai --pickle claude/local-agenda-claude-opus.pkl
python eval_implement.py --llm-config aws --pickle claude/local-agenda-claude-opus.pkl
```

With overrides:
```bash
python eval_implement.py --llm-config vllm --llm-override "model=my-finetuned-model" --pickle ...
```

**Option 2: Simple model name (`--model`)**
```bash
python eval_implement.py --model gpt-4o --pickle claude/local-agenda-claude-opus.pkl
python eval_implement.py --model claude-3-5-sonnet-20241022 --pickle ...
```

### All Options

```
usage: eval_implement.py [-h] (--model MODEL | --llm-config LLM_CONFIG)
                         [--llm-override LLM_OVERRIDE] [--llm-key LLM_KEY]
                         --pickle PICKLE [--success-only]
                         [--num-examples NUM_EXAMPLES] [--seed SEED]
                         [--output OUTPUT] [--verbose]
                         [--wandb] [--wandb-project PROJECT] [--wandb-run-name NAME]

Options:
  --model MODEL           LLM model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)
  --llm-config LLM_CONFIG Hydra config name or path to YAML file
  --llm-override          Hydra-style overrides (e.g., "model=gpt-4-turbo")
  --llm-key LLM_KEY       Which LLM from config: "code" or "write" (default: code)

  --pickle PICKLE         Path to agenda checkpoint pickle (can specify multiple)
  --success-only          Only evaluate on examples where Claude succeeded
  --num-examples N        Number of examples to evaluate (default: all)
  --seed SEED             Random seed for shuffling (default: 42)

  --output FILE           Path to save results JSON (default: implement_eval_results.json)
  --verbose               Enable verbose logging

  --no-wandb              Disable W&B logging (enabled by default)
  --wandb-project         W&B project name (default: formal-disco-implement)
  --wandb-run-name        W&B run name
```

## Evaluation Modes

### Mode 1: Success-Only (Fair Comparison)

```bash
python eval_implement.py --llm-config vllm --pickle ... --success-only
```

Only tests on examples where Claude succeeded (353 from Opus). This is the fair comparison:
- If fine-tuned model matches Claude's success rate → training preserved capability
- If fine-tuned model beats Claude → training improved capability
- If fine-tuned model is worse → training hurt capability

### Mode 2: All Examples (Ceiling Test)

```bash
python eval_implement.py --llm-config vllm --pickle ... --num-examples 500
```

Tests on all examples, including ones Claude failed. This shows if the model can do better than Claude on hard problems.

## Output Format

Results are saved as JSON:

```json
{
  "config": {
    "pickle_files": ["claude/local-agenda-claude-opus.pkl"],
    "success_only": true,
    "num_examples": 50,
    "seed": 42
  },
  "summary": {
    "total": 50,
    "outcome_counts": {
      "SUCCESS": 12,
      "GOAL_UNPROVEN": 8,
      "FAIL": 28,
      "ERROR": 2
    },
    "success_rate": 0.24,
    "gt_comparison": {
      "improved": 0,
      "same": 45,
      "worse": 5
    }
  },
  "results": [
    {
      "example_id": "distil/example_abc123.json",
      "idea": "A verified calculator service...",
      "generated_code": "class Calculator { ... }",
      "verification_outcome": "SUCCESS",
      "ground_truth_outcome": "success",
      "error": null
    },
    ...
  ]
}
```

## Interpreting Results

### Verification Outcomes

| Outcome | Meaning |
|---------|---------|
| `SUCCESS` | Dafny verified the program completely |
| `GOAL_UNPROVEN` | Code compiles but some proofs don't go through |
| `FAIL` | Syntax errors, type errors, or verification failures |
| `ERROR` | LLM call or Dafny execution failed |

### Ground Truth Comparison

The eval compares each result to what Claude achieved:

| Comparison | Meaning |
|------------|---------|
| `improved` | Model succeeded where Claude failed |
| `same` | Model achieved same outcome as Claude |
| `worse` | Model failed where Claude succeeded |

### What to Expect

**Baseline (vanilla model):**
- Should roughly match Claude's success rate on success-only examples
- May do worse on all examples (harder problems)

**Fine-tuned model:**
- If same or better than vanilla → training didn't hurt
- If worse than vanilla → training hurt (this is the red flag)

## Dependencies

The script imports from existing `formal-disco` modules:
- `prompt.py` - `system_implement()`, `format_implement_user()`
- `code_output_parser.py` - Extract code from markdown fences
- `dafny.py` - `DafnyProgram`, `VerificationOutcome`

Additional requirements:
```
langchain-core
langchain-openai    # for OpenAI models
langchain-anthropic # for Claude models
omegaconf           # for config loading
hydra-core          # for config instantiation
tqdm
```

## Programmatic Usage

```python
from eval_implement import DafnyImplementer, load_llm_from_config, load_implement_examples

# Load LLM
llm = load_llm_from_config("vllm")

# Load examples
examples = load_implement_examples(
    ["claude/local-agenda-claude-opus.pkl"],
    success_only=True,
)

# Create implementer
implementer = DafnyImplementer(llm, verbose=True)

# Evaluate a single example
result = implementer.evaluate(examples[0])
print(f"Outcome: {result.verification_outcome}")
print(f"Ground truth: {result.ground_truth_outcome}")
print(f"Generated code:\n{result.generated_code[:500]}")
```

## Comparison with eval_fixer.py

| Aspect | eval_implement (this) | eval_fixer |
|--------|----------------------|------------|
| Task | Implement from idea | Repair broken code |
| Data source | Claude pickles | DafnyBench |
| Output format | Full Dafny code | Diff |
| Ground truth | Claude's outcome | Verified program |
| Use case | Eval Claude-trained models | Eval repair capability |

## Workflow: Comparing Vanilla vs Fine-tuned

```bash
# 1. Eval vanilla model
python eval_implement.py \
    --llm-config vllm \
    --pickle claude/local-agenda-claude-opus.pkl \
    --success-only \
    --num-examples 100 \
    --output vanilla_implement.json

# 2. Switch to fine-tuned model (update vllm config or use override)
python eval_implement.py \
    --llm-config vllm \
    --llm-override "model=my-finetuned-model" \
    --pickle claude/local-agenda-claude-opus.pkl \
    --success-only \
    --num-examples 100 \
    --output finetuned_implement.json

# 3. Compare
python -c "
import json
with open('vanilla_implement.json') as f:
    v = json.load(f)
with open('finetuned_implement.json') as f:
    f = json.load(f)
print(f'Vanilla success rate: {v[\"summary\"][\"success_rate\"]:.1%}')
print(f'Fine-tuned success rate: {f[\"summary\"][\"success_rate\"]:.1%}')
"
```
