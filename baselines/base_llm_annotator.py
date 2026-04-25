#!/usr/bin/env python3
"""
Baseline LLM annotator for Verus-Bench.

Evaluates a base LLM (e.g. Claude Opus 4.6) on the Verus fixer task
WITHOUT the custom diff scaffolding used by verus_fixer. The model
receives the broken program and verifier output and produces a full
corrected program directly.

Runs pass@k with multiple iterations per attempt, outputting a JSON file.

Usage:
    python baselines/base_llm_annotator.py \
        --tasks-jsonl data/verus-bench/tasks.jsonl \
        --output results/baseline_opus.json \
        --llm-config aws-opus-4.5 \
        --k 4 --max-iters 3

    python baselines/base_llm_annotator.py \
        --tasks-jsonl data/verus-bench/tasks.jsonl \
        --output results/baseline_sonnet.json \
        --model claude-sonnet-4-6-20250514 \
        --k 4 --max-iters 3
"""

import argparse
import json
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code_output_parser import CodeOutputParser
from distill_common import remove_hints_verus
from eval_common import add_llm_args, create_llm_from_args
from language import Language, Program, VerificationOutcome
from tasks import _to_langchain_messages

_backend = Language.VERUS.get_backend()
logger = logging.getLogger(__name__)


def load_verusbench(jsonl_path: str, min_hints: int = 0) -> list[dict]:
    with open(jsonl_path) as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    examples = []
    for task in tasks:
        ground_truth = task.get("ground_truth")
        if not ground_truth:
            continue

        stripped, n_hints = remove_hints_verus(ground_truth, min_hints=min_hints)
        if n_hints < min_hints:
            continue

        prog = Program(stripped, Language.VERUS, name=task.get("task_id", ""))
        ver = prog.verify()
        notes = f"Verus verification output:\nstdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}"

        examples.append({
            "task_id": task.get("task_id", ""),
            "source": task.get("source", ""),
            "task_path": task.get("task_path", ""),
            "program": stripped,
            "notes": notes,
            "verification_outcome": ver.outcome.name,
            "ground_truth": ground_truth,
        })

    return examples


def _verify(program_text: str) -> dict:
    prog = Program(program_text, Language.VERUS, name="repaired")
    ver = prog.verify()
    return {
        "notes": f"Verus verification output:\nstdout:\n{ver.stdout}\n\nstderr:\n{ver.stderr}",
        "outcome": ver.outcome.name,
        "success": ver.outcome == VerificationOutcome.SUCCESS,
    }


def run_one_iteration(
    chain, program: str, notes: str, verification_outcome: str,
) -> dict:
    msgs = _to_langchain_messages(
        _backend.prompt_builder.repair_full(program=program, notes=notes)
    )

    try:
        response = chain.invoke(msgs).strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return {
            "program": program,
            "verification_outcome": verification_outcome,
            "notes": notes,
            "response": f"Error: {e}",
            "new_notes": "",
            "new_verification_outcome": "ERROR",
            "success": False,
        }

    try:
        ver = _verify(response)
    except Exception as e:
        logger.warning(f"Verification failed: {e}")
        return {
            "program": program,
            "verification_outcome": verification_outcome,
            "notes": notes,
            "response": response,
            "new_notes": f"Error: {e}",
            "new_verification_outcome": "ERROR",
            "success": False,
        }

    return {
        "program": program,
        "verification_outcome": verification_outcome,
        "notes": notes,
        "response": response,
        "new_notes": ver["notes"],
        "new_verification_outcome": ver["outcome"],
        "success": ver["success"],
    }


def evaluate_one(chain, example: dict, k: int, max_iters: int) -> dict:
    attempts = []
    any_success = False

    for _ in range(k):
        iteration_log = []
        current_program = example["program"]
        current_notes = example["notes"]
        current_outcome = example["verification_outcome"]

        if current_outcome == "SUCCESS":
            any_success = True
            attempts.append(iteration_log)
            continue

        for _ in range(max_iters):
            result = run_one_iteration(
                chain, current_program, current_notes, current_outcome,
            )
            iteration_log.append(result)

            if result["success"]:
                any_success = True
                break

            if (
                result["new_verification_outcome"] != "ERROR"
                and result["response"]
                and not result["response"].startswith("Error:")
            ):
                current_program = result["response"]
                current_notes = result["new_notes"]
                current_outcome = result["new_verification_outcome"]

        attempts.append(iteration_log)

    return {
        "task_id": example["task_id"],
        "source": example["source"],
        "task_path": example["task_path"],
        "success": any_success,
        "attempts": attempts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Baseline LLM annotator for Verus-Bench",
    )
    add_llm_args(parser)
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--min-hints", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    llm, model_desc = create_llm_from_args(args)
    logger.info(f"Using LLM: {model_desc}")
    chain = llm | CodeOutputParser()

    logger.info(f"Loading tasks from {args.tasks_jsonl}")
    examples = load_verusbench(args.tasks_jsonl, min_hints=args.min_hints)
    logger.info(f"Loaded {len(examples)} tasks")

    if args.num_examples is not None:
        random.seed(args.seed)
        random.shuffle(examples)
        examples = examples[:args.num_examples]
        logger.info(f"Using {len(examples)} examples (seed={args.seed})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = [None] * len(examples)
    success_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_one, chain, ex, args.k, args.max_iters): i
            for i, ex in enumerate(examples)
        }

        with tqdm(total=len(examples), desc="Evaluating") as pbar:
            for future in as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Task {i} failed: {e}")
                    result = {
                        "task_id": examples[i]["task_id"],
                        "source": examples[i]["source"],
                        "task_path": examples[i]["task_path"],
                        "success": False,
                        "attempts": [],
                        "error": str(e),
                    }
                results[i] = result
                if result.get("success"):
                    success_count += 1
                pbar.update(1)
                pbar.set_postfix(success=f"{success_count}/{sum(1 for r in results if r)}")

                with open(output_path, "w") as f:
                    json.dump([r for r in results if r], f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    n = len(results)
    logger.info(f"Done: {success_count}/{n} solved ({success_count/n*100:.1f}%)")


if __name__ == "__main__":
    main()
