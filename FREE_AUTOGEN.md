# Free Autogen Distillation Pipeline

Generate distillation data from verified Dafny programs by systematic mutation.

## Overview

Unlike VFP (which empties entire lemma bodies), this approach:
1. Takes fully verified programs
2. Removes one invariant/assertion at a time
3. Captures the Dafny error for the broken version
4. Creates repair examples to restore the removed line

This produces more targeted training data with specific, informative error messages.

## Prerequisites

- Verified Dafny files in `../dafny-free-autogen/`

## Usage

### List removable lines

See what can be mutated in each file:

```bash
python free_autogen_distill.py list --max-files 20
```

### Generate distillation pickle

```bash
# Full generation with Dafny errors
python free_autogen_distill.py generate --output free_autogen.pkl

# Fast generation (skip Dafny, use placeholder errors)
python free_autogen_distill.py generate --output free_autogen.pkl --skip-dafny
```

Options:
- `--input-dir`: Directory with .dfy files (default: `../dafny-free-autogen`)
- `--output`, `-o`: Output pickle file (default: `free_autogen.pkl`)
- `--max-files`: Limit number of files to process
- `--skip-dafny`: Skip running Dafny for faster generation

### Check stats and train

```bash
python distill.py stats -d free_autogen.pkl
python distill.py sft data=free_autogen.pkl output_dir=free-autogen-sft-out
```

## What gets removed

The script identifies and removes:
- `invariant` clauses in loops
- `assert` statements
- `decreases` clauses

Each removal creates one training example if the removal causes verification to fail.

## Example

Given this verified program:
```dafny
method SumToN(n: nat) returns (sum: nat)
  ensures sum == n * (n + 1) / 2
{
  sum := 0;
  var i := 0;
  while i < n
    invariant 0 <= i <= n
    invariant sum == i * (i + 1) / 2  // <-- removed
  {
    i := i + 1;
    sum := sum + i;
  }
}
```

Removing the second invariant produces:
- **Input**: Broken program + Dafny error ("postcondition could not be proved")
- **Output**: Diff to add `+ invariant sum == i * (i + 1) / 2`

## Advantages over VFP approach

1. **Specific errors**: Dafny points to exactly what's wrong (loop invariant, postcondition)
2. **Targeted fixes**: Model learns to add one specific line, not write entire proofs
3. **Matches real usage**: This is how verification debugging actually works
