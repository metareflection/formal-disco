#!/usr/bin/env python3
"""
Shared utilities for evaluation scripts.

Provides:
- LLM creation (create_llm, load_llm_from_config, create_llm_from_args)
- Argparse helpers (add_llm_args, add_eval_args)
- WandB helpers (init_wandb, finish_wandb)
- Pickle example loading (load_pickle_examples)
- Results helpers (print_summary, save_results)
"""

import argparse
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any, Optional

from distill_common import get_content

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM creation
# ---------------------------------------------------------------------------

def create_llm(model_name: str, temperature: float = 0.0):
    """
    Create a LangChain LLM instance based on the model name.

    Supports:
    - OpenAI models (gpt-4o, gpt-4-turbo, etc.)
    - Anthropic models (claude-3-5-sonnet-*, etc.)
    """
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif model_name.startswith("claude-"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Supported prefixes: gpt-*, o1*, o3*, claude-*")


def load_llm_from_config(
    config_name: str,
    overrides: Optional[list[str]] = None,
    llm_key: str = "code",
):
    """
    Load a LangChain LLM from a Hydra config file.

    Args:
        config_name: Either a config name (e.g., "openai", "aws", "vllm")
                     or a path to a YAML file (e.g., "config/llm/openai.yaml")
        overrides: Optional list of Hydra overrides (e.g., ["model=gpt-4-turbo"])
        llm_key: Which LLM to use from the config ("code" or "write")

    Returns:
        A LangChain LLM instance
    """
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
        raise KeyError(
            f"LLM key '{llm_key}' not found in config. Available: {available_keys}"
        )

    llm_cfg = cfg[llm_key]
    logger.info(f"Loading LLM from config: {config_name} (key={llm_key})")
    logger.info(f"LLM config: {OmegaConf.to_yaml(llm_cfg)}")

    return instantiate(llm_cfg)


def create_llm_from_args(args) -> tuple:
    """
    Create LLM from parsed CLI args.

    Returns:
        (llm, model_desc) tuple
    """
    if args.model:
        llm = create_llm(args.model, getattr(args, 'temperature', 0.0))
        model_desc = args.model
    else:
        llm = load_llm_from_config(
            args.llm_config,
            overrides=args.llm_override if args.llm_override else None,
            llm_key=getattr(args, 'llm_key', 'code'),
        )
        model_desc = f"{args.llm_config}:{getattr(args, 'llm_key', 'code')}"
    return llm, model_desc


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------

def add_llm_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard LLM CLI arguments to a parser."""
    llm_group = parser.add_mutually_exclusive_group(required=True)
    llm_group.add_argument('--model', type=str,
                           help='LLM model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)')
    llm_group.add_argument('--llm-config', type=str,
                           help='Hydra LLM config name (e.g., openai, aws, vllm) or path to YAML file')
    parser.add_argument('--llm-override', type=str, action='append', default=[],
                        help='Hydra-style overrides for LLM config (e.g., "model=gpt-4-turbo")')
    parser.add_argument('--llm-key', type=str, default='code',
                        help='Which LLM to use from config: "code" or "write" (default: code)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='LLM temperature (only used with --model)')


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add standard evaluation output/logging arguments."""
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging (enabled by default)')


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------

def init_wandb(args, project: str, config_extras: dict = None) -> bool:
    """Initialize wandb if available and not disabled. Returns use_wandb bool."""
    use_wandb = not args.no_wandb and WANDB_AVAILABLE
    if not args.no_wandb and not WANDB_AVAILABLE:
        logger.warning("wandb not installed, skipping logging. Install with: pip install wandb")
    if use_wandb:
        config = {}
        if config_extras:
            config.update(config_extras)
        wandb.init(project=project, config=config)
    return use_wandb


def finish_wandb(use_wandb: bool, success_count: int, total: int) -> None:
    """Log final summary and finish wandb run."""
    if use_wandb:
        wandb.summary["final_success_count"] = success_count
        wandb.summary["final_success_rate"] = success_count / total if total > 0 else 0
        wandb.summary["total_examples"] = total
        wandb.finish()


# ---------------------------------------------------------------------------
# Pickle example loading
# ---------------------------------------------------------------------------

def load_pickle_examples(
    pickle_paths: list[str],
    prompt_types: list[str],
    seed: int = 42,
    success_only: bool = False,
) -> list[dict]:
    """
    Load distill-examples from pickle files, filtering by prompt type.

    Args:
        pickle_paths: List of paths to pkl files
        prompt_types: e.g. ['repair'], ['implement'], ['lemma_synth', 'repair']
        seed: Random seed for shuffling
        success_only: Only include examples with outcome='success'

    Returns:
        List of parsed JSON dicts from the distill-examples.
        Each dict has an added '_path' key with the original object path.
    """
    examples = []

    for pkl_path in pickle_paths:
        logger.info(f"Loading from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

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
            if ex.get('prompt') not in prompt_types:
                continue
            if success_only and ex.get('outcome') != 'success':
                continue
            ex['_path'] = path
            examples.append(ex)

    random.seed(seed)
    random.shuffle(examples)

    logger.info(f"Loaded {len(examples)} examples (prompt_types={prompt_types})")
    return examples


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def print_summary(task_name: str, success_count: int, total: int) -> None:
    """Print a standard evaluation summary."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {success_count}/{total} {task_name}")
    print(f"Success rate: {100*success_count/total:.1f}%" if total > 0 else "No examples")
    print(f"{'='*60}")


def save_results(output_path: str, results: list, metadata: dict) -> None:
    """Save evaluation results to JSON."""
    output_data = {**metadata, "results": results}
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to {output_path}")
