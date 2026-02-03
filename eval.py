#!/usr/bin/env python3
"""
Unified evaluation entry point.

Usage:
    python eval.py task=fixer llm=openai
    python eval.py task=fixer llm=vllm data=fixer_val
    python eval.py task=lemma_synth llm=openai data=lemma_val num_examples=50
    python eval.py task=implement llm=vllm data=implement_val
    python eval.py task=fixer llm=vllm data=glob glob="../dafny-vfp/autogen/bench*minimized/**/*.dfy"
"""

import json
import logging
import random
import sys

from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from eval_common import print_summary, save_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@hydra_main(config_path="config", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    # Instantiate the task
    task = instantiate(cfg.task)
    logger.info(f"Task: {task.name}")

    # Instantiate the LLM
    llm = instantiate(cfg.llm.code)
    logger.info(f"LLM: {OmegaConf.to_yaml(cfg.llm.code)}")

    # Extract examples from the data sources
    sources = OmegaConf.to_container(cfg.data.sources, resolve=True)
    examples = task.extract_examples(sources)

    # Apply seed and limit
    if cfg.seed is not None:
        random.seed(cfg.seed)
        random.shuffle(examples)

    if cfg.num_examples is not None:
        examples = examples[:cfg.num_examples]

    logger.info(f"Evaluating on {len(examples)} examples")

    if not examples:
        logger.error("No examples to evaluate!")
        return

    # Initialize wandb
    use_wandb = cfg.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Run evaluation
    outcome = task.evaluate(
        llm, examples,
        use_wandb=use_wandb,
        verbose=cfg.verbose,
    )

    # Print summary
    print_summary(
        f"{task.name} successes",
        outcome["success_count"],
        outcome["total"],
    )

    # Save results
    if cfg.output:
        save_results(cfg.output, outcome["results"], {
            "task": task.name,
            "num_examples": outcome["total"],
            "success_count": outcome["success_count"],
            "success_rate": outcome["success_count"] / outcome["total"] if outcome["total"] > 0 else 0,
        })

    # Finish wandb
    if use_wandb:
        wandb.summary["final_success_count"] = outcome["success_count"]
        wandb.summary["final_success_rate"] = (
            outcome["success_count"] / outcome["total"] if outcome["total"] > 0 else 0
        )
        wandb.summary["total_examples"] = outcome["total"]
        wandb.finish()


if __name__ == "__main__":
    main()
