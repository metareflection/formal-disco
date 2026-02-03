#!/usr/bin/env python3
"""
Unified data extraction entry point.

Extracts training/validation examples from data sources using task-specific logic,
then writes them as pickle files compatible with distill.py sft.

Usage:
    # Extract fixer examples from verified programs (default data=agenda)
    python extract.py task=fixer output_prefix=fixer

    # Extract lemma examples
    python extract.py task=lemma_synth output_prefix=lemma

    # Extract from a specific pickle
    python extract.py task=fixer 'data.sources=[{type: pickle, path: run3.pkl, extract_from_verified: true}]' output_prefix=fixer

    # Extract implement examples from .dfy files
    python extract.py task=implement 'data.sources=[{type: dfy, glob: "autogen/**/*.dfy"}]' output_prefix=implement
"""

import json
import logging
import random
import sys
from pathlib import Path

from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from distill_common import create_agenda_pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@hydra_main(config_path="config", config_name="extract", version_base=None)
def main(cfg: DictConfig) -> None:
    task = instantiate(cfg.task)
    logger.info(f"Task: {task.name}")

    output_prefix = cfg.get("output_prefix", task.name)
    val_fraction = cfg.get("val_fraction", 0.2)
    seed = cfg.get("seed", 42)

    # Extract examples from the data sources
    sources = OmegaConf.to_container(cfg.data.sources, resolve=True)
    examples = task.extract_examples(sources)
    logger.info(f"Extracted {len(examples)} examples")

    if not examples:
        logger.error("No examples extracted!")
        return

    # Split by program path if available, else random split
    random.seed(seed)
    program_paths = sorted(set(
        ex.get('metadata', {}).get('program_path', f'__idx_{i}')
        for i, ex in enumerate(examples)
    ))
    random.shuffle(program_paths)
    val_count = max(1, int(len(program_paths) * val_fraction))
    val_paths = set(program_paths[:val_count])

    train_examples = [
        ex for i, ex in enumerate(examples)
        if ex.get('metadata', {}).get('program_path', f'__idx_{i}') not in val_paths
    ]
    val_examples = [
        ex for i, ex in enumerate(examples)
        if ex.get('metadata', {}).get('program_path', f'__idx_{i}') in val_paths
    ]

    for ex in train_examples:
        ex.setdefault('metadata', {})['split'] = 'train'
    for ex in val_examples:
        ex.setdefault('metadata', {})['split'] = 'val'

    # Write pickles
    prefix = f"distil/{task.prompt_type}"

    train_path = Path(f"{output_prefix}_train.pkl")
    val_path = Path(f"{output_prefix}_val.pkl")

    if train_examples:
        create_agenda_pickle(train_examples, train_path, prefix=prefix)
        print(f"Saved {len(train_examples)} train examples to {train_path}")

    if val_examples:
        create_agenda_pickle(val_examples, val_path, prefix=prefix)
        print(f"Saved {len(val_examples)} val examples to {val_path}")
    else:
        print("Warning: no val examples generated")

    # Summary
    print(f"\n{'='*50}")
    print(f"EXTRACTION SUMMARY ({task.name})")
    print(f"{'='*50}")
    print(f"  total: {len(examples)}")
    print(f"  train: {len(train_examples)}")
    print(f"  val:   {len(val_examples)}")


if __name__ == "__main__":
    main()
