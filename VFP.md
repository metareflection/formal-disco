# VFP Distillation Pipeline

Generate distillation data from the VFP (Verified Function Programming) benchmark for training fixer models.

## Prerequisites

- VFP autogen repository at `../dafny-vfp-autogen/`: https://github.com/metareflection/dafny-vfp-autogen
- dafny-tasker at `../dafny-tasker/`

## Step 1: Generate Emptied JSON

Note: this step is already checked in.

Use dafny-tasker to create the emptied JSON from VFP benchmark files:

```bash
cd ../dafny-tasker && python -m dafny_tasker.cli empty \
    --inputs '../dafny-vfp-autogen/bench*-minimized/*dfy' \
    --out ../dafny-vfp-autogen/vfp_autogen_emptied_minimized.json
```

This creates a JSON file where each entry has:
- `id`: Identifier like `{file}_{method}_empty`
- `type`: "empty"
- `program`: The program with one lemma/method body emptied
- `output`: The lemma body that should be filled in (empty string for trivial lemmas)

## Step 2: Generate Distillation Pickle

```bash
# Full generation (runs Dafny to get verification errors)
python vfp_distill.py generate --output vfp_distill.pkl

# Fast generation (skip Dafny, use placeholder errors)
python vfp_distill.py generate --output vfp_distill.pkl --skip-dafny

# Specify custom JSON path
python vfp_distill.py generate \
    --json-path ../dafny-vfp-autogen/vfp_autogen_emptied_minimized.json \
    --output vfp_distill.pkl
```

Options:
- `--json-path`: Path to emptied JSON (default: `../dafny-vfp-autogen/vfp_autogen_emptied_minimized.json`)
- `--output`, `-o`: Output pickle file (default: `vfp_distill.pkl`)
- `--max-programs`: Limit number of programs to process
- `--skip-dafny`: Skip running Dafny for faster generation
- `--include-trivial`: Include entries with empty body (skipped by default)

## Step 3: Verify Diffs (Optional)

Verify that generated diffs correctly insert the lemma bodies:

```bash
python vfp_distill.py verify --max-programs 50
python vfp_distill.py verify --verbose  # Show details on failures
```

## Step 4: Check Stats

```bash
python distill.py stats -d vfp_distill.pkl
```

## Step 5: Train

```bash
python distill.py sft data=vfp_distill.pkl output_dir=vfp-sft-out
```

## Notes

- **Dataset size**: ~150 non-trivial examples from 1553 total entries (most have empty proofs that verify trivially)
- Trivial lemmas (empty `output` field) are skipped by default since they verify without any proof body
- The diff format uses the lemma declaration as a unique anchor, then navigates to `{` and inserts the body
- Example diff:
  ```
  @@lemma CompleteInduction(n: nat, P: nat -> bool)@@
  =   requires P(0)
  =   requires forall k: nat :: k > 0 && (forall j: nat :: j < k ==> P(j)) ==> P(k)
  =   ensures P(n)
  = {
  +   if n == 0 {
  +     // Base case
  +   } else {
  +     CompleteInduction(n-1, P);
  +   }
  ```
