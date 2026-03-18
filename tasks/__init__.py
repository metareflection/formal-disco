#!/usr/bin/env python3
"""
Evaluation task framework.

Provides an EvaluationTask base class that unifies:
  (a) extracting training/test examples from data sources
  (b) evaluating a model on the task
  (c) generating SFT training records

Add a new task by subclassing EvaluationTask in a new file under tasks/.
"""

import glob as globmod
import importlib
import inspect
import json
import logging
import pickle
import pkgutil
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def _to_langchain_messages(messages):
    """Convert a list of ChatMessage dicts to LangChain BaseMessage objects."""
    mapping = {"system": SystemMessage, "user": HumanMessage}
    return [mapping[m["role"]](content=m["content"]) for m in messages]


def _evaluate_one_worker(args):
    """Top-level helper so ProcessPoolExecutor can pickle it."""
    task, llm, example = args
    return task.evaluate_one(llm, example)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EvaluationTask(ABC):
    """Base class for all evaluation/training tasks."""

    # Subclasses must set these.
    name: str           # e.g. "fixer", "implement", "lemma_synth"
    prompt_type: str    # matches distill example schema: "repair", "implement", "lemma_synth"

    # ------------------------------------------------------------------
    # (a) Data extraction
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_examples(self, sources: list[dict]) -> list[dict]:
        """Extract examples from the given data sources.

        Each source is a dict like:
          {"type": "pickle", "path": "...", "prompt_types": [...]}
          {"type": "dfy", "glob": "DafnyBench/**/*.dfy"}

        Sources are passed in from the `data` config group (config/data/*.yaml),
        not stored on the task itself.

        Returns list of DistillExample dicts.
        """
        ...

    # ------------------------------------------------------------------
    # (b) Evaluation
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate_one(self, llm: Any, example: dict) -> dict:
        """Evaluate model on one example.

        Returns result dict with at least 'success': bool.
        """
        ...

    # ------------------------------------------------------------------
    # (c) Training records
    # ------------------------------------------------------------------

    @abstractmethod
    def to_training_record(self, example: dict) -> dict | None:
        """Convert a distill example to an SFT chat record.

        Returns {"messages": <list[ChatMessage]>, "completion": str}
        or None to skip this example.
        """
        ...

    # ------------------------------------------------------------------
    # Shared evaluation runner
    # ------------------------------------------------------------------

    def evaluate(
        self,
        llm: Any,
        examples: list[dict],
        use_wandb: bool = False,
        verbose: bool = False,
    ) -> dict:
        """Run evaluate_one over all examples in parallel, aggregate metrics, log."""
        N_PROCS = 32

        results = [None] * len(examples)
        success_count = 0

        with ThreadPoolExecutor(max_workers=N_PROCS) as executor:
            futures = {
                executor.submit(_evaluate_one_worker, (self, llm, ex)): i
                for i, ex in enumerate(examples)
            }

            with tqdm(total=len(examples), desc=f"Evaluating {self.name}") as pbar:
                for future in as_completed(futures):
                    i = futures[future]
                    result = future.result()
                    results[i] = result
                    if result.get('success'):
                        success_count += 1
                    pbar.update(1)

                    if use_wandb and WANDB_AVAILABLE:
                        n = sum(1 for r in results if r is not None)
                        wandb.log({
                            "examples_evaluated": n,
                            "success_count": success_count,
                            "success_rate": success_count / n,
                        })

        return {
            "results": results,
            "success_count": success_count,
            "total": len(results),
        }

    # ------------------------------------------------------------------
    # Retry helper for iterative tasks (fixer, lemma_synth)
    # ------------------------------------------------------------------

    def evaluate_with_retries(
        self,
        llm: Any,
        example: dict,
        max_attempts: int = 3,
        verbose: bool = False,
    ) -> dict:
        """Shared generate-verify-retry loop.

        Subclasses that use this must override:
          generate(llm, example) -> response
          apply_response(example, response) -> program_text
          verify(program_text) -> result dict with 'success', 'stdout', 'stderr'
          update_for_retry(example, result) -> updated example
        """
        result = {"success": False, "num_attempts": 0}

        for attempt in range(max_attempts):
            if verbose:
                logger.info(f"Attempt {attempt + 1}/{max_attempts}")

            try:
                response = self.generate(llm, example)
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
                continue

            try:
                program_text = self.apply_response(example, response)
            except Exception as e:
                logger.warning(f"Failed to apply response: {e}")
                continue

            result = self.verify(program_text)
            result["num_attempts"] = attempt + 1

            if result.get("success"):
                return result

            example = self.update_for_retry(example, result)

        result["num_attempts"] = max_attempts
        return result

    # Hooks for evaluate_with_retries — override in subclasses that use it.
    def generate(self, llm: Any, example: dict) -> str:
        raise NotImplementedError

    def apply_response(self, example: dict, response: str) -> str:
        raise NotImplementedError

    def verify(self, program_text: str) -> dict:
        raise NotImplementedError

    def update_for_retry(self, example: dict, result: dict) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared data loaders
# ---------------------------------------------------------------------------

def load_pickle_source(source: dict) -> list[dict]:
    """Load examples from a pickle source.

    source keys:
      path: str — path to pickle file
      prompt_types: list[str] — filter by prompt type
      success_only: bool — only successful examples (default False)
    """
    from distill_common import get_content

    pkl_path = source["path"]
    prompt_types = source.get("prompt_types", [])
    success_only = source.get("success_only", False)

    logger.info(f"Loading pickle: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    examples = []
    for path, obj in data.get('objects', {}).items():
        if obj.type not in ('distil-example', 'distill-example'):
            continue
        content = get_content(obj)
        if not content:
            continue
        try:
            ex = json.loads(content)
        except Exception:
            continue
        if prompt_types and ex.get('prompt') not in prompt_types:
            continue
        if success_only and ex.get('outcome') != 'success':
            continue
        ex['_path'] = path
        examples.append(ex)

    logger.info(f"Loaded {len(examples)} examples from {pkl_path}")
    return examples


def load_dfy_source(source: dict) -> list[tuple[str, str]]:
    """Load .dfy files matching a glob pattern.

    source keys:
      glob: str — glob pattern for .dfy files

    Returns list of (name, text) tuples.
    """
    pattern = source["glob"]
    files = sorted(globmod.glob(pattern, recursive=True))

    programs = []
    for filepath in files:
        p = Path(filepath)
        try:
            text = p.read_text()
            programs.append((p.stem, text))
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")

    logger.info(f"Loaded {len(programs)} .dfy files from {pattern}")
    return programs


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def discover_tasks() -> dict[str, type[EvaluationTask]]:
    """Auto-discover all EvaluationTask subclasses in the tasks package."""
    tasks_dir = Path(__file__).parent
    found = {}

    for info in pkgutil.iter_modules([str(tasks_dir)]):
        if info.name.startswith('_'):
            continue
        try:
            module = importlib.import_module(f"tasks.{info.name}")
        except Exception as e:
            logger.warning(f"Failed to import tasks.{info.name}: {e}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, EvaluationTask) and obj is not EvaluationTask:
                task_name = getattr(obj, 'name', info.name)
                found[task_name] = obj

    return found
