#!/usr/bin/env python3

"""Distillation utilities.

Currently supports:
    python distill.py stats -d <agenda_pickle>
    python distill.py sft  (see config/distill.yaml)

stats prints summary statistics for objects written under the `distil/` prefix
in an agenda pickled checkpoint, produced by `LocalAgenda`'s checkpointing.

sft trains an SFT model from distillation examples stored in an agenda pickle.
"""

import argparse
import json
import pickle
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from hydra import main as hydra_main
from omegaconf import DictConfig


DEFAULT_HF_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


DISTIL_PREFIX = "distil/"


@dataclass(frozen=True)
class DistillStats:
    total: int
    outcomes: Counter[str]
    by_prompt_total: dict[str, int]
    by_prompt_outcomes: dict[str, Counter[str]]


def _iter_distil_objects(pickle_data: dict[str, Any]) -> Iterable[tuple[str, Any]]:
    """Yield (path, obj) pairs for agenda objects whose paths start with DISTIL_PREFIX."""
    objects = pickle_data.get("objects", {})
    # Stored objects are instances of agenda.Object (dataclass).
    for path, obj in objects.items():
        if isinstance(path, str) and path.startswith(DISTIL_PREFIX):
            yield path, obj


def _parse_distil_json(obj: Any) -> dict[str, Any] | None:
    """Parse a distillation example object into a dict, or return None if invalid."""
    content = getattr(obj, "content", None)
    if content is None:
        return None

    if isinstance(content, bytes):
        raw = content.decode("utf-8", errors="replace")
    else:
        raw = str(content)

    try:
        return json.loads(raw)
    except Exception:
        return None


def compute_distil_stats(pickle_path: str | Path) -> DistillStats:
    p = Path(pickle_path)
    with p.open("rb") as f:
        data = pickle.load(f)

    total = 0
    outcomes: Counter[str] = Counter()
    by_prompt_total: dict[str, int] = defaultdict(int)
    by_prompt_outcomes: dict[str, Counter[str]] = defaultdict(Counter)

    for _, obj in _iter_distil_objects(data):
        ex = _parse_distil_json(obj)
        if not ex:
            continue

        prompt = str(ex.get("prompt", "unknown"))
        outcome = str(ex.get("outcome", "unknown"))

        total += 1
        outcomes[outcome] += 1
        by_prompt_total[prompt] += 1
        by_prompt_outcomes[prompt][outcome] += 1

    return DistillStats(
        total=total,
        outcomes=outcomes,
        by_prompt_total=dict(by_prompt_total),
        by_prompt_outcomes={k: Counter(v) for k, v in by_prompt_outcomes.items()},
    )


def _print_stats(stats: DistillStats) -> None:
    print(f"distil objects: {stats.total}")

    print("\nby outcome:")
    if stats.total == 0:
        print("  (none)")
    else:
        for outcome, n in stats.outcomes.most_common():
            print(f"  {outcome}: {n}")

    print("\nby prompt:")
    if not stats.by_prompt_total:
        print("  (none)")
        return

    for prompt in sorted(stats.by_prompt_total.keys()):
        total = stats.by_prompt_total[prompt]
        print(f"  {prompt}: {total}")
        for outcome, n in stats.by_prompt_outcomes[prompt].most_common():
            print(f"    {outcome}: {n}")


def _cmd_stats(args: argparse.Namespace) -> None:
    stats = compute_distil_stats(args.data)
    _print_stats(stats)


def _iter_distill_examples_from_pickle(pickle_path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(pickle_path)
    with p.open("rb") as f:
        data = pickle.load(f)

    for _, obj in _iter_distil_objects(data):
        ex = _parse_distil_json(obj)
        if ex:
            yield ex


def build_sft_records(
    pickle_path: str | Path,
    success_only: bool,
    outcome_success_values: tuple[str, ...] = ("success",),
) -> list[dict[str, str]]:
    """Build TRL/HF records for chat-style SFT.

    Output schema:
      {"messages": <list[dict(role, content)]>, "completion": <str>}

    We reconstruct system+user messages from stored arguments using `prompt.py`.
    """
    from prompt import reconstruct_chat_messages
    from patch import TEXT_BEFORE_EXAMPLE, TEXT_DIFF_EXAMPLE, TEXT_AFTER_EXAMPLE

    records: list[dict[str, str]] = []

    for ex in _iter_distill_examples_from_pickle(pickle_path):
        kind = str(ex.get("prompt", "unknown"))
        args = ex.get("arguments")
        if not isinstance(args, dict):
            continue

        outcome = ex.get("outcome")
        outcome_s = "" if outcome is None else str(outcome).lower()

        if success_only and outcome_s not in outcome_success_values:
            continue

        response = ex.get("response")
        if response is None:
            continue
        response_s = str(response)

        try:
            messages = reconstruct_chat_messages(
                kind,
                args,
                example_before=TEXT_BEFORE_EXAMPLE,
                example_diff=TEXT_DIFF_EXAMPLE,
                example_after=TEXT_AFTER_EXAMPLE,
            )
        except Exception:
            continue

        records.append({"messages": json.dumps(messages, ensure_ascii=False), "completion": response_s})

    return records


def _train_with_trl(
    records: list[dict[str, str]],
    model_id: str,
    output_dir: str,
    max_seq_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: float,
    warmup_ratio: float,
    lr_scheduler_type: str,
    logging_steps: int,
    save_steps: int,
    seed: int,
    use_wandb: bool,
    wandb_project: str | None,
    wandb_run_name: str | None,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str] | None,
) -> None:
    """Train an SFT model with TRL SFTTrainer + PEFT LoRA.

    Notes:
    - Multi-GPU is handled by Accelerate/Trainer when launched via `accelerate launch`.
    - We keep the implementation conservative: no custom packing tricks, and we train on a
      prompt+completion schema.
    """
    import os

    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", wandb_project or "formal-disco")
        if wandb_run_name:
            os.environ.setdefault("WANDB_NAME", wandb_run_name)
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        # Trainer will pick this up.
        os.environ.setdefault("WANDB_WATCH", "false")

    if not records:
        raise ValueError("No records to train on (after filtering).")

    ds = Dataset.from_list(records)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        # Common for decoder-only models.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )

    report_to = ["wandb"] if use_wandb else ["none"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(per_device_train_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        learning_rate=float(learning_rate),
        num_train_epochs=float(num_train_epochs),
        warmup_ratio=float(warmup_ratio),
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=int(logging_steps),
        save_steps=int(save_steps),
        save_total_limit=2,
        seed=int(seed),
        report_to=report_to,
        remove_unused_columns=False,
        fp16=False,
        bf16=True,
    )

    def _formatting_func(ex: dict[str, Any]) -> str:
        messages = json.loads(ex["messages"])
        # Apply model's default chat template.
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_text + ex["completion"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        peft_config=peft_config,
        dataset_text_field=None,
        max_seq_length=int(max_seq_length),
        args=training_args,
        formatting_func=_formatting_func,
    )

    trainer.train()

    # Save adapter + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Merge LoRA into base model weights and save a merged checkpoint for inference.
    merged_dir = str(Path(output_dir) / "merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)


def _main_sft() -> None:
    """Hydra entrypoint for `distill.py sft`.

    We keep `stats` as argparse-based for convenience, but training gets messy without Hydra.
    """

    @dataclass
    class SftCfg:
        data: str = "local-agenda.pkl"
        model_id: str = DEFAULT_HF_MODEL_ID
        output_dir: str = "sft-out"
        success_only: bool = True
        # Treat GOAL_UNPROVEN as success? (optional)
        treat_goal_unproven_as_success: bool = False

        # Training
        max_seq_length: int = 4096
        per_device_train_batch_size: int = 1
        gradient_accumulation_steps: int = 4
        learning_rate: float = 2e-4
        num_train_epochs: float = 1.0
        warmup_ratio: float = 0.03
        lr_scheduler_type: str = "cosine"
        logging_steps: int = 10
        save_steps: int = 200
        seed: int = 0

        # LoRA
        lora_r: int = 16
        lora_alpha: int = 32
        lora_dropout: float = 0.05
        lora_target_modules: list[str] | None = None

        # W&B
        wandb: bool = True
        wandb_project: str | None = None
        wandb_run_name: str | None = None

    @hydra_main(config_path="config", config_name="distill", version_base=None)
    def run(cfg: DictConfig) -> None:
        # Merge user cfg into our typed defaults (simple, no schema enforcement)
        c = SftCfg(**{k: v for k, v in cfg.items()})

        success_values = ("success",)
        if c.treat_goal_unproven_as_success:
            success_values = ("success", "goal_unproven")

        records = build_sft_records(
            pickle_path=c.data,
            success_only=bool(c.success_only),
            outcome_success_values=success_values,
        )

        # If running under Hydra, output_dir is relative to the hydra run dir.
        out_dir = os.path.abspath(c.output_dir)

        _train_with_trl(
            records=records,
            model_id=c.model_id or DEFAULT_HF_MODEL_ID,
            output_dir=out_dir,
            max_seq_length=c.max_seq_length,
            per_device_train_batch_size=c.per_device_train_batch_size,
            gradient_accumulation_steps=c.gradient_accumulation_steps,
            learning_rate=c.learning_rate,
            num_train_epochs=c.num_train_epochs,
            warmup_ratio=c.warmup_ratio,
            lr_scheduler_type=c.lr_scheduler_type,
            logging_steps=c.logging_steps,
            save_steps=c.save_steps,
            seed=c.seed,
            use_wandb=c.wandb,
            wandb_project=c.wandb_project,
            wandb_run_name=c.wandb_run_name,
            lora_r=c.lora_r,
            lora_alpha=c.lora_alpha,
            lora_dropout=c.lora_dropout,
            lora_target_modules=c.lora_target_modules,
        )

    run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="distill.py")
    sub = parser.add_subparsers(dest="command", required=True)

    stats_p = sub.add_parser("stats", help="Summarize distillation objects in an agenda pickle")
    stats_p.add_argument("-d", "--data", required=True, help="Path to agenda checkpoint pickle")
    stats_p.set_defaults(func=_cmd_stats)

    # `sft` is Hydra-based.
    sft_p = sub.add_parser("sft", help="Train an SFT model from distil examples (Hydra config)")
    sft_p.set_defaults(func=lambda _args: _main_sft())

    ns = parser.parse_args(argv)
    ns.func(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
5