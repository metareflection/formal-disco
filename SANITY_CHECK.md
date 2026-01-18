# Distillation Pipeline Sanity Check

This document describes how to verify the distillation pipeline is working correctly by training on DafnyBench and evaluating on the same data.

## Overview

The sanity check:
1. Generates synthetic distillation data from DafnyBench (hints_removed → ground_truth)
2. Trains a model on this data
3. Evaluates on the same DafnyBench programs

If the pipeline works, the model should perform very well since it trained on the answers. If it performs poorly, there's a bug in the pipeline.

## Step 1: Generate Synthetic Data

On a login node (no GPU needed):

```bash
# Generate synthetic distillation data from DafnyBench
# --skip-dafny skips running Dafny verification (faster, fine for sanity check)
python sanity_check_distill.py generate --skip-dafny --output sanity_check.pkl

# Verify the output
python3 -c "
import sys; sys.path.insert(0, '.')
from sanity_check_distill import Object
import pickle, json
with open('sanity_check.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'Generated {len([p for p in data[\"objects\"] if p.startswith(\"distil/\")])} examples')
"
```

Expected output: ~580 examples.

## Step 2: Train the Model

Submit a training job (requires GPU).

First, edit `config/distill.yaml` to point to your sanity check data:

```yaml
# In config/distill.yaml, change:
data: /path/to/your/sanity_check.pkl   # absolute path to generated pickle
output_dir: sanity-sft-out
num_train_epochs: 1.0
wandb: false  # or true if you want logging
```

Then run:

```bash
python distill.py sft
```

Example SLURM script (`train_sanity.sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=sanity-sft
#SBATCH --output=sanity-sft-%j.out
#SBATCH --error=sanity-sft-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4:00:00

cd $SLURM_SUBMIT_DIR

# Activate your environment
source /path/to/your/venv/bin/activate

# Make sure config/distill.yaml is configured first!
python distill.py sft
```

Submit with: `sbatch train_sanity.sbatch`

## Step 3: Serve the Model with vLLM

After training completes, serve the merged model:

```bash
# Interactive GPU session for serving
salloc --gpus=1 --mem=80G --time=2:00:00

# Serve the merged model
vllm serve sanity-sft-out/merged \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 16384

# Or if you need the model name to match vllm.yaml config:
vllm serve sanity-sft-out/merged \
    --served-model-name qwen3-coder-flash \
    --port 8000
```

Keep this running in one terminal/session.

## Step 4: Run Evaluation

In another terminal (same node, or set VLLM_BASE_URL appropriately):

```bash
# Set the vLLM endpoint
export VLLM_BASE_URL=http://localhost:8000/v1

# Extract the nontrivial programs used in training (~479 programs)
python sanity_check_subset.py sanity_check.pkl -o sanity_programs.txt

# Run evaluation on only those programs
python eval_fixer.py \
    --llm-config vllm \
    --programs-file sanity_programs.txt \
    --max-attempts 3 \
    --output sanity_eval_results.json
```

This ensures you evaluate on exactly the same programs the model was trained on.

## Step 5: Interpret Results

**Expected (pipeline working):**
- Success rate should be **high** (>70%) since the model trained on these exact problems
- The model should produce diffs that closely match the training data

**If success rate is low (<30%):**

1. **Check chat template mismatch:**
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("sanity-sft-out/merged")
   messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "hello"}]
   print(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
   ```
   Compare this with what vLLM produces.

2. **Check model loading:**
   ```bash
   # Verify the merged model exists and has reasonable size
   ls -lh sanity-sft-out/merged/
   ```

3. **Check vLLM logs** for any errors or warnings about chat templates.

4. **Test vLLM directly:**
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen3-coder-flash",
       "messages": [{"role": "user", "content": "Hello"}],
       "max_tokens": 50
     }'
   ```

## Quick Reference

| Step | Command | GPU? |
|------|---------|------|
| Generate data | `python sanity_check_distill.py generate --skip-dafny -o sanity_check.pkl` | No |
| Extract programs | `python sanity_check_subset.py sanity_check.pkl -o sanity_programs.txt` | No |
| Train | Edit `config/distill.yaml`, then `python distill.py sft` | Yes |
| Serve | `vllm serve sanity-sft-out/merged --port 8000` | Yes |
| Eval | `python eval_fixer.py --llm-config vllm --programs-file sanity_programs.txt` | No (needs vLLM running) |

## Debugging Tips

### Compare with baseline
Run the same eval with the base model (no fine-tuning) to establish a baseline:
```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000
python eval_fixer.py --llm-config vllm --num-programs 100 --output baseline_results.json
```

### Check training loss
If using wandb, check that training loss decreased. If not using wandb, check the training logs.

### Inspect model outputs
Run with `--verbose` to see the actual diffs the model produces:
```bash
python eval_fixer.py --llm-config vllm --num-programs 5 --verbose
```
