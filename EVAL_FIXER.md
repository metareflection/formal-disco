# eval_fixer.py

Evaluation script for the LLM Fixer on DafnyBench.

This script faithfully extracts the core repair logic from `worker/llm_fixer.py` (without the Agenda/Task abstractions) and evaluates it on adding annotations to Dafny programs from the DafnyBench benchmark.

## Overview

The fixer works by:
1. Taking a Dafny program that fails verification
2. Prompting an LLM with the program + verification errors
3. LLM produces a **diff** (not a full program replacement)
4. Applying the diff to the program
5. Re-verifying and iterating up to `max_attempts` times

This diff-based approach is different from other annotation methods (like dafny-annotator) which insert annotations at various positions.

## DafnyBench Setup

The script needs the DafnyBench dataset. By default it looks for `DafnyBench/` in the current directory.

**Option 1: Clone into current directory**
```bash
cd formal-disco
git clone https://github.com/sun-wendy/DafnyBench.git
python eval_fixer.py --model gpt-4o --fraction 0.1
```

**Option 2: Point to existing location**
```bash
python eval_fixer.py --model gpt-4o --benchmark-path /path/to/DafnyBench
python eval_fixer.py --model gpt-4o --benchmark-path ../dafny-annotator/DafnyBench
```

The script tries several directory structures automatically:
- `<benchmark-path>/DafnyBench/dataset/hints_removed/`
- `<benchmark-path>/dataset/hints_removed/`
- `<benchmark-path>/hints_removed/`

## Caching

The script caches verification outcomes to avoid re-verifying all programs on each run.

**How it works:**
1. First run: verifies each program to check if it's trivial (already verifies), saves results to cache
2. Subsequent runs: reads from cache, skips verification

**Cache location:** `.fixer_outcome_cache.json` (configurable via `--cache-path`)

```bash
# First run - slow (verifies 782 programs to filter trivial ones)
python eval_fixer.py --model gpt-4o --fraction 0.1

# Second run - fast (uses cached verification outcomes)
python eval_fixer.py --model gpt-4o --fraction 0.1

# Use a different cache file
python eval_fixer.py --model gpt-4o --cache-path my_cache.json

# Skip filtering entirely (run on all programs, even trivial ones)
python eval_fixer.py --model gpt-4o --no-filter
```

**Cache format:**
```json
{
  "Clover_binary_search_no_hints": "GOAL_UNPROVEN",
  "Clover_bubble_sort_no_hints": "FAIL",
  "Clover_abs_no_hints": "SUCCESS"
}
```

Programs with `"SUCCESS"` are filtered out (they're trivial - already verify without fixes).

## Dependencies

The script imports from existing `formal-disco` modules:
- `prompt.py` - `system_repair()`, `format_repair_user()`
- `patch.py` - `apply_text_diff()`, diff examples
- `code_output_parser.py` - Extract code from markdown fences
- `dafny.py` - `DafnyProgram`, `VerificationOutcome`

Additional requirements:
```
langchain-core
langchain-openai    # for OpenAI models
langchain-anthropic # for Claude models
langchain-aws       # for Bedrock models
omegaconf           # for config loading
hydra-core          # for config instantiation
tqdm
```

## Usage

### Quick Testing (10% of dataset)

```bash
python eval_fixer.py --llm-config openai --fraction 0.1
python eval_fixer.py --model gpt-4o --fraction 0.1
```

### Full Evaluation

```bash
# Using a model name directly
python eval_fixer.py --model gpt-4o --num-programs 100 --output results.json

# Using Hydra config
python eval_fixer.py --llm-config aws --num-programs 100 --output results.json
```

### LLM Configuration

**Option 1: Simple model name (`--model`)**
```bash
python eval_fixer.py --model gpt-4o
python eval_fixer.py --model claude-3-5-sonnet-20241022
python eval_fixer.py --model o1-preview
```

**Option 2: Hydra config (`--llm-config`)**

Uses configs from `config/llm/`:
```bash
python eval_fixer.py --llm-config openai   # Uses config/llm/openai.yaml
python eval_fixer.py --llm-config aws      # Uses config/llm/aws.yaml
python eval_fixer.py --llm-config vllm     # Uses config/llm/vllm.yaml
```

With overrides:
```bash
python eval_fixer.py --llm-config openai --llm-override "model=gpt-4-turbo"
python eval_fixer.py --llm-config vllm --llm-override "model=my-finetuned-model"
```

Use `write` LLM instead of `code`:
```bash
python eval_fixer.py --llm-config openai --llm-key write
```

**Option 3: Custom YAML file**
```bash
python eval_fixer.py --llm-config path/to/my-llm.yaml
```

### All Options

```
usage: eval_fixer.py [-h] (--model MODEL | --llm-config LLM_CONFIG)
                     [--llm-override LLM_OVERRIDE] [--llm-key LLM_KEY]
                     [--benchmark-path BENCHMARK_PATH]
                     [--num-programs NUM_PROGRAMS] [--fraction FRACTION]
                     [--max-attempts MAX_ATTEMPTS] [--output OUTPUT]
                     [--cache-path CACHE_PATH] [--temperature TEMPERATURE]
                     [--verbose] [--skip SKIP] [--no-filter]

Options:
  --model MODEL           LLM model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)
  --llm-config LLM_CONFIG Hydra config name or path to YAML file
  --llm-override          Hydra-style overrides (e.g., "model=gpt-4-turbo")
  --llm-key LLM_KEY       Which LLM from config: "code" or "write" (default: code)

  --benchmark-path        Path to DafnyBench directory (default: DafnyBench)
  --num-programs N        Number of programs to evaluate
  --fraction F            Fraction of dataset (e.g., 0.1 for 10%)
  --max-attempts N        Maximum repair attempts per program (default: 3)

  --output FILE           Path to save results JSON
  --cache-path FILE       Verification outcome cache (default: .fixer_outcome_cache.json)
  --temperature T         LLM temperature (default: 0.0, only with --model)
  --verbose               Enable verbose logging
  --skip N                Skip first N programs (for train/test split)
  --no-filter             Don't filter out trivially-verified programs
  --programs-file FILE    File with program names to evaluate (one per line)
```

## Focused Evaluation on a Subset

When iterating on a new model or approach, running on the full benchmark is slow. A useful workflow is to:

1. Run a baseline evaluation
2. Extract a subset of programs (e.g., successes + same number of failures)
3. Run the new approach on that consistent subset

This lets you quickly compare approaches on the same set of problems.

### Step 1: Run Baseline

```bash
python eval_fixer.py --model gpt-4o --output baseline.json
```

### Step 2: Extract Subset

Use `eval_fixer_extract_subset.py` to extract program names from the baseline results:

```bash
# Successes + same number of failures (default)
python eval_fixer_extract_subset.py baseline.json > subset.txt

# Only successful programs
python eval_fixer_extract_subset.py baseline.json --successes-only > subset.txt

# Only failed programs
python eval_fixer_extract_subset.py baseline.json --failures-only > subset.txt

# Successes + half as many failures
python eval_fixer_extract_subset.py baseline.json --failure-ratio 0.5 > subset.txt
```

### Step 3: Run New Approach on Subset

```bash
python eval_fixer.py --model claude-3-5-sonnet-20241022 --programs-file subset.txt --output new_approach.json
```

### Comparing Results

Both JSON files will have results for the same programs, making comparison straightforward:

```python
import json

with open('baseline.json') as f:
    baseline = {r['program_name']: r['success'] for r in json.load(f)['results']}

with open('new_approach.json') as f:
    new = {r['program_name']: r['success'] for r in json.load(f)['results']}

# Programs where new approach succeeded but baseline failed
improved = [p for p in new if new[p] and not baseline.get(p)]

# Programs where baseline succeeded but new approach failed
regressed = [p for p in baseline if baseline[p] and not new.get(p)]

print(f"Improved: {len(improved)}, Regressed: {len(regressed)}")
```

## Programmatic Usage

```python
from eval_fixer import DafnyFixer, load_llm_from_config, load_dafnybench_programs

# Load LLM from config
llm = load_llm_from_config("aws", overrides=["model_id=anthropic.claude-3-opus"])

# Or use any LangChain LLM directly
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create fixer
fixer = DafnyFixer(llm, max_attempts=3, verbose=True)

# Fix a single program
result = fixer.fix(program_text, program_name="my_program")
print(f"Success: {result.success}")
print(f"Attempts: {result.num_attempts}")
print(f"Final program:\n{result.final_program}")

# Load and process DafnyBench programs
programs = load_dafnybench_programs("DafnyBench", use_hints_removed=True)
for name, text in programs[:10]:
    result = fixer.fix(text, program_name=name)
    print(f"{name}: {'✓' if result.success else '✗'}")
```

## Output Format

Results are saved as JSON:

```json
{
  "model": "gpt-4o",
  "llm_config": null,
  "llm_overrides": [],
  "max_attempts": 3,
  "num_programs": 100,
  "success_count": 42,
  "success_rate": 0.42,
  "results": [
    {
      "program_name": "Clover_binary_search_no_hints",
      "success": true,
      "num_attempts": 2,
      "verification_outcome": "SUCCESS",
      "original_program": "...",
      "final_program": "...",
      "diffs_applied": ["@@ ...", "@@ ..."]
    },
    ...
  ]
}
```

## How It Works

The `DafnyFixer` class is a faithful extraction of `worker/llm_fixer.py`:

1. **Prompt Construction** (same as `llm_fixer.py:45-61`)
   ```python
   ChatPromptTemplate.from_messages([
       ("system", system_repair(example_before=..., example_diff=..., example_after=...)),
       ("human", format_repair_user(program=..., notes=...)),
   ])
   ```

2. **Chain** (same as `llm_fixer.py:65`)
   ```python
   chain = prompt | llm | CodeOutputParser()
   ```

3. **Repair Loop** (same as `llm_fixer.py:67-178`)
   - Build prompt with program + verification output
   - Call LLM to get a diff
   - Apply diff with `apply_text_diff()`
   - Verify with `DafnyProgram.verify()`
   - Repeat until success or max attempts

## Diff Format

The LLM produces diffs in a simple line-based format:

```
@@
= existing line to find
+ new line to add after
@@
= another anchor
- line to delete
+ replacement line
```

Directives:
- `@@` - Anchor/search marker
- `= line` - Find and advance past this line
- `- line` - Delete this line
- `+ line` - Insert this line

## Available LLM Configs

| Config | Provider | Default Model |
|--------|----------|---------------|
| `openai` | OpenAI | gpt-4o-mini (write), gpt-5-mini (code) |
| `aws` | AWS Bedrock | claude-3-haiku (write), claude-3-5-sonnet (code) |
| `vllm` | vLLM (local) | qwen3-coder-flash |
| `awsbest` | AWS Bedrock | (see config) |
| `ollama` | Ollama (local) | (see config) |

## Comparison with dafny-annotator

| Aspect | eval_fixer (this) | dafny-annotator |
|--------|-------------------|-----------------|
| Output format | Diff | Individual annotations |
| Insertion strategy | LLM decides via diff | Try all valid positions |
| Search | Iterative repair | Greedy search over positions |
| Feedback | Verification errors | Verification outcome |
| LLM backend | LangChain (any) | vLLM |
