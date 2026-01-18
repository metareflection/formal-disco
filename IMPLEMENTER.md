# Implementer Distillation Pipeline

Generate distillation data for the implementer worker (idea → verified program).

## Overview

Takes verified Dafny programs that have "Idea prompt:" comments and creates training examples for generating implementations from specifications.

## Prerequisites

- Verified Dafny files with idea prompts in `../dafny-free-autogen/`: https://github.com/metareflection/dafny-free-autogen

## Usage

### List idea prompts

```bash
python implementer_distill.py list --max-files 20
```

### Generate distillation pickle

```bash
python implementer_distill.py generate --output implementer.pkl
```

Options:
- `--input-dir`: Directory with .dfy files (default: `../dafny-free-autogen`)
- `--output`, `-o`: Output pickle file (default: `implementer.pkl`)
- `--max-files`: Limit number of files to process

### Train

```bash
python distill.py stats -d implementer.pkl
python distill.py sft data=implementer.pkl output_dir=implementer-sft-out
```

## Example

Input idea:
```
Write a function that returns the maximum of two integers with a postcondition ensuring the result is >= both inputs.
```

Output program:
```dafny
method Max(a: int, b: int) returns (max: int)
  ensures max >= a && max >= b
  ensures max == a || max == b
{
  if a >= b {
    max := a;
  } else {
    max := b;
  }
}
```

## Notes

- 100 examples from dafny-free-autogen
- Each file must have a `// Idea prompt: ...` comment
- The idea comment is removed from the output program
