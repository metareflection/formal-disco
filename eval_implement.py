#!/usr/bin/env python3
"""
Evaluation script for the implement task on Claude-generated data.

This script evaluates a model's ability to implement Dafny programs from ideas,
using the same distribution as the Claude training data.

Usage:
    # Using a Hydra LLM config
    python eval_implement.py --llm-config vllm --pickle claude/local-agenda-claude-opus.pkl --num-examples 50

    # Filter to only examples that succeeded in training data
    python eval_implement.py --llm-config vllm --pickle claude/local-agenda-claude-opus.pkl --success-only

    # Multiple pickle files
    python eval_implement.py --llm-config vllm --pickle claude/local-agenda-claude.pkl --pickle claude/local-agenda-claude-opus.pkl
"""

import os
import json
import argparse
import logging
import random
import pickle
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from langchain_core.prompts import ChatPromptTemplate
from code_output_parser import CodeOutputParser
from prompt import system_implement, format_implement_user
from dafny import DafnyProgram, VerificationOutcome

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ImplementResult:
    """Result of a single implement evaluation."""
    example_id: str
    idea: str
    generated_code: str
    verification_outcome: str
    verification_stdout: str = ""
    verification_stderr: str = ""
    ground_truth_outcome: Optional[str] = None  # What Claude achieved
    error: Optional[str] = None


def load_implement_examples(
    pickle_paths: list[str],
    success_only: bool = False,
    seed: int = 42,
) -> list[dict]:
    """
    Load implement examples from pickle files.

    Args:
        pickle_paths: List of paths to agenda checkpoint pickles
        success_only: If True, only load examples where Claude succeeded
        seed: Random seed for shuffling

    Returns:
        List of dicts with 'id', 'idea', 'ground_truth_outcome', 'ground_truth_response'
    """
    examples = []

    for pickle_path in pickle_paths:
        logger.info(f"Loading from {pickle_path}")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        objects = data.get('objects', {})

        for path, obj in objects.items():
            if obj.type != 'distill-example':
                continue
            if not obj.content:
                continue

            try:
                ex = json.loads(obj.content.decode('utf-8'))
            except Exception:
                continue

            if ex.get('prompt') != 'implement':
                continue

            outcome = ex.get('outcome', 'unknown')
            if success_only and outcome != 'success':
                continue

            args = ex.get('arguments', {})
            idea = args.get('idea', '')
            if not idea:
                continue

            examples.append({
                'id': path,
                'idea': idea,
                'ground_truth_outcome': outcome,
                'ground_truth_response': ex.get('response', ''),
            })

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(examples)

    logger.info(f"Loaded {len(examples)} implement examples")
    return examples


def create_llm(model_name: str, temperature: float = 0.0):
    """Create a LangChain LLM instance based on the model name."""
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif model_name.startswith("claude-"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def load_llm_from_config(
    config_name: str,
    overrides: Optional[list[str]] = None,
    llm_key: str = "code",
):
    """Load a LangChain LLM from a Hydra config file."""
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    overrides = overrides or []

    if config_name.endswith(".yaml") or config_name.endswith(".yml"):
        config_path = Path(config_name)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_name}")
        cfg = OmegaConf.load(config_path)
    else:
        config_dir = Path(__file__).parent / "config" / "llm"
        config_path = config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"LLM config '{config_name}' not found. "
                f"Available configs: {[p.stem for p in config_dir.glob('*.yaml')]}"
            )
        cfg = OmegaConf.load(config_path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        if llm_key in cfg:
            cfg[llm_key] = OmegaConf.merge(cfg[llm_key], override_cfg)

    if llm_key not in cfg:
        available_keys = list(cfg.keys())
        raise KeyError(f"LLM key '{llm_key}' not found in config. Available: {available_keys}")

    llm_cfg = cfg[llm_key]
    logger.info(f"Loading LLM from config: {config_name} (key={llm_key})")
    return instantiate(llm_cfg)


class DafnyImplementer:
    """Evaluates implement task: idea -> Dafny code -> verify."""

    def __init__(
        self,
        llm: Any,
        verbose: bool = False,
    ) -> None:
        self._llm = llm
        self._verbose = verbose

        # Build prompt for implement task
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", system_implement()),
            ("user", "{user_message}"),
        ])

        self._chain = self._prompt | self._llm | CodeOutputParser()

    def evaluate(self, example: dict) -> ImplementResult:
        """
        Evaluate a single implement example.

        Args:
            example: Dict with 'id', 'idea', 'ground_truth_outcome'

        Returns:
            ImplementResult with verification outcome
        """
        example_id = example['id']
        idea = example['idea']
        ground_truth_outcome = example.get('ground_truth_outcome')

        # Format the user message
        user_message = format_implement_user(idea=idea)

        # Call LLM
        try:
            generated_code = self._chain.invoke({"user_message": user_message}).strip()
        except Exception as e:
            logger.warning(f"LLM call failed for {example_id}: {e}")
            return ImplementResult(
                example_id=example_id,
                idea=idea,
                generated_code="",
                verification_outcome="ERROR",
                ground_truth_outcome=ground_truth_outcome,
                error=str(e),
            )

        if self._verbose:
            logger.info(f"Generated code for {example_id}:\n{generated_code[:500]}...")

        # Verify with Dafny
        try:
            prog = DafnyProgram(generated_code, name=example_id)
            ver = prog.verify()
            outcome = ver.outcome.name
            stdout = ver.stdout
            stderr = ver.stderr
        except Exception as e:
            logger.warning(f"Verification failed for {example_id}: {e}")
            return ImplementResult(
                example_id=example_id,
                idea=idea,
                generated_code=generated_code,
                verification_outcome="ERROR",
                ground_truth_outcome=ground_truth_outcome,
                error=str(e),
            )

        if self._verbose:
            logger.info(f"Verification outcome for {example_id}: {outcome}")

        return ImplementResult(
            example_id=example_id,
            idea=idea,
            generated_code=generated_code,
            verification_outcome=outcome,
            verification_stdout=stdout,
            verification_stderr=stderr,
            ground_truth_outcome=ground_truth_outcome,
        )


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate implement task on Claude-generated data'
    )

    # LLM configuration
    llm_group = parser.add_mutually_exclusive_group(required=True)
    llm_group.add_argument('--model', type=str,
                           help='LLM model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)')
    llm_group.add_argument('--llm-config', type=str,
                           help='Hydra LLM config name (e.g., openai, aws, vllm)')

    parser.add_argument('--llm-override', type=str, action='append', default=[],
                        help='Hydra-style overrides for LLM config')
    parser.add_argument('--llm-key', type=str, default='code',
                        help='Which LLM to use from config (default: code)')

    # Data configuration
    parser.add_argument('--pickle', type=str, action='append', required=True,
                        help='Path to agenda checkpoint pickle (can specify multiple)')
    parser.add_argument('--success-only', action='store_true',
                        help='Only evaluate on examples where Claude succeeded')
    parser.add_argument('--num-examples', type=int, default=None,
                        help='Number of examples to evaluate (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')

    # Output configuration
    parser.add_argument('--output', type=str, default='implement_eval_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    # Wandb
    parser.add_argument('--wandb', action='store_true',
                        help='Log to Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='formal-disco-implement',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name')

    args = parser.parse_args()

    # Initialize wandb if requested
    if args.wandb:
        if not WANDB_AVAILABLE:
            logger.warning("wandb not installed, skipping logging")
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )

    # Load LLM
    if args.model:
        llm = create_llm(args.model)
    else:
        llm = load_llm_from_config(
            args.llm_config,
            overrides=args.llm_override,
            llm_key=args.llm_key,
        )

    # Load examples
    examples = load_implement_examples(
        args.pickle,
        success_only=args.success_only,
        seed=args.seed,
    )

    if args.num_examples:
        examples = examples[:args.num_examples]

    logger.info(f"Evaluating on {len(examples)} examples")

    # Create implementer
    implementer = DafnyImplementer(llm, verbose=args.verbose)

    # Run evaluation
    results = []
    outcome_counts = {'SUCCESS': 0, 'GOAL_UNPROVEN': 0, 'FAIL': 0, 'ERROR': 0}
    gt_comparison = {'improved': 0, 'same': 0, 'worse': 0}

    for example in tqdm(examples, desc="Evaluating"):
        result = implementer.evaluate(example)
        results.append(result)

        outcome_counts[result.verification_outcome] = outcome_counts.get(result.verification_outcome, 0) + 1

        # Compare to ground truth
        gt = result.ground_truth_outcome
        pred = result.verification_outcome.lower()
        if gt:
            outcome_order = {'success': 2, 'goal_unproven': 1, 'fail': 0, 'error': -1, 'unknown': -1}
            gt_score = outcome_order.get(gt.lower(), -1)
            pred_score = outcome_order.get(pred, -1)
            if pred_score > gt_score:
                gt_comparison['improved'] += 1
            elif pred_score == gt_score:
                gt_comparison['same'] += 1
            else:
                gt_comparison['worse'] += 1

        # Log to wandb
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({
                'success_rate': outcome_counts['SUCCESS'] / len(results),
                'goal_unproven_rate': outcome_counts['GOAL_UNPROVEN'] / len(results),
                'fail_rate': outcome_counts['FAIL'] / len(results),
            })

    # Print summary
    total = len(results)
    print("\n" + "=" * 50)
    print("IMPLEMENT EVAL RESULTS")
    print("=" * 50)
    print(f"Total examples: {total}")
    print(f"\nVerification outcomes:")
    for outcome, count in sorted(outcome_counts.items()):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    print(f"\nComparison to Claude ground truth:")
    for comp, count in gt_comparison.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {comp}: {count} ({pct:.1f}%)")

    # Save results
    output_data = {
        'config': {
            'pickle_files': args.pickle,
            'success_only': args.success_only,
            'num_examples': len(examples),
            'seed': args.seed,
        },
        'summary': {
            'total': total,
            'outcome_counts': outcome_counts,
            'success_rate': outcome_counts['SUCCESS'] / total if total > 0 else 0,
            'gt_comparison': gt_comparison,
        },
        'results': [
            {
                'example_id': r.example_id,
                'idea': r.idea[:500],  # Truncate for readability
                'generated_code': r.generated_code,
                'verification_outcome': r.verification_outcome,
                'ground_truth_outcome': r.ground_truth_outcome,
                'error': r.error,
            }
            for r in results
        ],
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to {args.output}")

    if args.wandb and WANDB_AVAILABLE:
        wandb.log({
            'final_success_rate': outcome_counts['SUCCESS'] / total if total > 0 else 0,
            'final_goal_unproven_rate': outcome_counts['GOAL_UNPROVEN'] / total if total > 0 else 0,
        })
        wandb.finish()


if __name__ == '__main__':
    main()
