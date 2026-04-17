#!/usr/bin/env python3
"""
Convert a Verus benchmark JSONL into a training pickle.

Works with any JSONL that has {task_id, task, ground_truth} entries,
e.g. Verus-Bench or VeruSAGE-Bench.

Each (task, ground_truth) pair becomes a repair distill example:
  - arguments.program = task code (unverified / empty-body)
  - arguments.notes = Verus verification errors on the task code
  - response = diff from task to ground_truth

Usage:
    python verusbench_to_pkl.py [--jsonl PATH] [--output PREFIX] [--verify]

    --jsonl   Path to tasks.jsonl (default: ../verus-proof-synthesis/benchmarks/Verus-Bench/tasks.jsonl)
    --output  Output prefix (default: derived from jsonl parent dir). Produces <prefix>_train.pkl and <prefix>_val.pkl
    --verify  Actually run Verus on the task code to get real error messages (slower but better)
"""

import argparse
import json
import logging
import random
from pathlib import Path

from distill_common import create_agenda_pickle
from patch import apply_text_diff, compute_text_diff

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Convert Verus benchmark JSONL to training pickle")
    parser.add_argument("--jsonl", default="../verus-proof-synthesis/benchmarks/Verus-Bench/tasks.jsonl")
    parser.add_argument("--output", default=None, help="Output prefix (default: derived from jsonl parent dir name)")
    parser.add_argument("--verify", action="store_true", help="Run Verus to get real error messages")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(args.jsonl).parent.name.lower().replace("-", "_")

    source_name = Path(args.jsonl).parent.name

    verifier_backend = None
    if args.verify:
        from language import Language, Program
        verifier_backend = Language.VERUS.get_backend()

    tasks = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    logger.info(f"Loaded {len(tasks)} tasks from {args.jsonl}")

    examples = []
    skipped = 0

    for task in tasks:
        unverified = task.get("task")
        verified = task.get("ground_truth")
        if not unverified or not verified:
            skipped += 1
            continue

        # Compute diff
        try:
            diff = compute_text_diff(unverified, verified)
            # Sanity check: applying the diff should recover the verified version
            reconstructed = apply_text_diff(unverified, diff)
            if reconstructed != verified:
                logger.warning(f"Diff roundtrip failed for {task.get('task_id', '?')}, skipping")
                skipped += 1
                continue
        except Exception as e:
            logger.warning(f"Failed to compute diff for {task.get('task_id', '?')}: {e}")
            skipped += 1
            continue

        # Get verification errors
        if verifier_backend:
            from language import Program, Language
            prog = Program(unverified, Language.VERUS, name=task.get("task_id", ""))
            ver = prog.verify()
            notes = f"Verus verification output:\nstdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}"
        else:
            notes = "(verification errors not available -- run with --verify to include them)"

        examples.append({
            "prompt": "repair",
            "arguments": {"program": unverified, "notes": notes},
            "response": diff,
            "outcome": "success",
            "language": "Verus",
            "metadata": {
                "source": source_name,
                "task_id": task.get("task_id", ""),
                "bench_source": task.get("source", ""),
            },
        })

    logger.info(f"Created {len(examples)} examples ({skipped} skipped)")

    # Train/val split
    random.seed(args.seed)
    random.shuffle(examples)
    val_count = max(1, int(len(examples) * args.val_fraction))
    val_examples = examples[:val_count]
    train_examples = examples[val_count:]

    train_path = Path(f"{args.output}_train.pkl")
    val_path = Path(f"{args.output}_val.pkl")

    if train_examples:
        create_agenda_pickle(train_examples, train_path, prefix="distil/repair")
        print(f"Saved {len(train_examples)} train examples to {train_path}")

    if val_examples:
        create_agenda_pickle(val_examples, val_path, prefix="distil/repair")
        print(f"Saved {len(val_examples)} val examples to {val_path}")

    print(f"\nTotal: {len(examples)}, Train: {len(train_examples)}, Val: {len(val_examples)}")


if __name__ == "__main__":
    main()
