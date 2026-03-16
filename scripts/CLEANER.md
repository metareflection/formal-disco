# cleaner.py - Fix Unverified Dafny Programs

A tool that automatically repairs failing Dafny programs by removing constructs that prevent verification.

## Quick Start

```bash
# Clean all FAIL programs in an agenda
python cleaner.py agenda.pkl -o agenda_cleaned.pkl

# Process a subset
python cleaner.py agenda.pkl -n 1000 -o cleaned.pkl
```

## How It Works

The cleaner uses a **reduction strategy**: systematically remove failing constructs until the program verifies.

### Removal Priority

Constructs are removed in order of "disposability":

1. **ensures** - Postconditions (removing makes verification easier)
2. **requires** - Preconditions
3. **lemma** - Helper lemmas (often the source of failures)
4. **function** - Functions with unproven properties
5. **method** - Last resort

### Algorithm

```
1. Verify program
2. If SUCCESS → done
3. Find all removable constructs
4. Try removing each (in priority order)
5. Accept if verification improves (FAIL → GOAL_UNPROVEN → SUCCESS)
6. Repeat until SUCCESS or no progress
```

## Performance

| Metric | Value |
|--------|-------|
| Success rate | **54%** |
| Runtime | ~30s per program |
| Avg lines removed | 113 |
| Avg lemmas removed | 2.1 |
| Avg functions removed | 3.1 |

## Usage

```
usage: cleaner.py [-h] [-o OUTPUT] [-n MAX] [-t TIMEOUT] [-v] input

positional arguments:
  input                 Input agenda pickle file

options:
  -o, --output OUTPUT   Output pickle file (default: input_cleaned.pkl)
  -n, --max MAX         Maximum programs to process
  -t, --timeout TIMEOUT Verification timeout in seconds (default: 15)
  -v, --verbose         Print details for each program
```

## Examples

### Clean entire agenda
```bash
python cleaner.py agenda-run3.pkl -o agenda-run3-cleaned.pkl
```

### Process in batches (for large agendas)
```bash
# First 1000
python cleaner.py agenda.pkl -n 1000 -o batch1.pkl

# Use output as input for next batch
python cleaner.py batch1.pkl -n 1000 -o batch2.pkl
```

### Verbose mode (see each program)
```bash
python cleaner.py agenda.pkl -n 100 -v -o cleaned.pkl
```

## Output

The cleaner writes to a new output file (default: `input_cleaned.pkl`).

"Cleaned to SUCCESS" means Dafny actually verified the program - it passes `dafny verify`.

The output pickle contains:
- Original SUCCESS programs (unchanged)
- Original GOAL_UNPROVEN programs (unchanged)
- FAIL programs that were cleaned to SUCCESS (updated content + status)
- FAIL programs that couldn't be cleaned (unchanged, still FAIL)

So the output is the full agenda with some FAIL programs fixed, not a filtered list of only verified programs.

### Sample Output

```
Loading agenda-run3.pkl...
Found 29889 FAIL programs
Processing first 100
Cleaning: 100%|██████████| 100/100 [53:58<00:00, 32.39s/it]

==================================================
CLEANING SUMMARY
==================================================
Total processed:         100
Cleaned to SUCCESS:      54 (54.0%)
Cleaned to GOAL_UNPROVEN:0
Still FAIL:              46

Applying 54 repairs...
Saving to agenda-run3-cleaned.pkl...
Done!
```

## Trade-offs

### What You Get
- Programs that verify ✓
- Valid training data ✓
- Automated bulk processing ✓

### What You Lose
- Some specifications (postconditions)
- Helper lemmas and proofs
- Test methods
- Overall "interestingness"

## When to Use

| Situation | Use cleaner.py? |
|-----------|-----------------|
| Need verified programs for training | ✓ Yes |
| Want to maximize verification count | ✓ Yes |
| Need to preserve all specifications | ✗ No, use llm_fixer.py |
| Programs have syntax errors | ✗ No, regenerate |

## Comparison with Other Tools

| Tool | Success Rate | Preserves Specs | Speed |
|------|--------------|-----------------|-------|
| **cleaner.py** | 54% | No | Fast |
| llm_fixer.py | ~30% | Yes | Slow |

## Parallelization

For large agendas, run multiple instances on different subsets:

```bash
# Terminal 1
python cleaner.py agenda.pkl -n 1000 -o part1.pkl

# Terminal 2 (after part1 done)
python cleaner.py part1.pkl -n 1000 -o part2.pkl
```

Or modify the script to use multiprocessing (the `clean_program` function is independent per program).

## Troubleshooting

### "Still FAIL after cleaning"
The program has fundamental issues beyond removable constructs. Options:
- Regenerate from the original idea
- Use `llm_fixer.py` to add missing proofs

### Slow processing
- Reduce `--timeout` (default 15s)
- Some programs trigger expensive verification

### Out of memory
- Process in smaller batches with `-n`
