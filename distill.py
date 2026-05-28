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
import math
import json
import pickle
import os
import sys

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

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


def _reconstruct_program(ex: dict[str, Any], language: str) -> str | None:
    """Reconstruct the program associated with a distill example, if any.

    - implement/initiate/generate: response *is* a full program.
    - repair/extend: response is a text diff to apply to arguments["program"].
    - lemma_synth (dafny): substitute the response body back into the hollowed
      program at the named lemma.
    - idea (and unknown kinds): no associated program; returns None.

    Returns None if the program can't be reconstructed for any reason; callers
    treat those as unrankable.
    """
    from patch import apply_text_diff

    kind = str(ex.get("prompt", "") or "")
    args = ex.get("arguments") or {}
    response = ex.get("response")
    if response is None or not isinstance(args, dict):
        return None
    response_s = str(response)

    if kind in ("implement", "initiate", "generate"):
        return response_s
    if kind in ("repair", "extend"):
        base = args.get("program", "")
        if not base:
            return None
        try:
            return apply_text_diff(str(base), response_s)
        except Exception:
            return None
    if kind == "lemma_synth" and language.lower() == "dafny":
        try:
            from tasks.lemma_synth import insert_lemma_body, extract_body_from_response
            base = args.get("program", "")
            lemma_name = args.get("lemma_name", "")
            if not base or not lemma_name:
                return None
            body = extract_body_from_response(response_s)
            return insert_lemma_body(str(base), str(lemma_name), body)
        except Exception:
            return None
    return None


def _select_top_surprisal_indices(
    programs: list[str | None],
    fraction: float,
    language: str,
) -> set[int] | None:
    """Pick the indices to keep within a single prompt-type group.

    `programs[i]` is the reconstructed program for the i-th example in the
    group, or None if it couldn't be reconstructed.

    For each rankable example, compute the per-metric maximum surprisal under
    the pooled feature distribution of the group. Rank examples within each
    metric (rank 1 = most surprising). Each example's score is the *best*
    (smallest) rank it achieves across any metric — i.e. examples that are top
    on any single feature are favored. We keep the top `fraction` by score.

    Returns the set of kept indices, or None if the group has no rankable
    examples (caller falls back to keeping the whole group).
    """
    from language import Language, Program

    backend = Language[language.upper()].get_backend()
    lang_enum = Language[language.upper()]

    pooled: dict[str, Counter] = defaultdict(Counter)
    feats_by_i: dict[int, dict[str, Counter]] = {}
    for i, p_text in enumerate(programs):
        if not p_text:
            continue
        try:
            fs = backend.feature_sets(Program(p_text, lang_enum))
        except Exception:
            continue
        feats_by_i[i] = fs
        for metric, c in fs.items():
            pooled[metric].update(c)

    if not feats_by_i:
        return None

    surprisal_by_i: dict[int, dict[str, float]] = {}
    for i, fs in feats_by_i.items():
        s: dict[str, float] = {}
        for metric, counter in fs.items():
            if not counter:
                continue
            pooled_c = pooled[metric]
            total = sum(pooled_c.values())
            if total == 0:
                continue
            best = 0.0
            for v in counter:
                c = pooled_c[v]
                if c == 0:
                    best = math.inf
                    break
                val = math.log2(total / c)
                if val > best:
                    best = val
            s[metric] = best
        if s:
            surprisal_by_i[i] = s

    if not surprisal_by_i:
        return None

    # Rank within each metric (descending surprisal -> rank 1, 2, ...).
    all_metrics = {m for s in surprisal_by_i.values() for m in s}
    best_rank_by_i: dict[int, int] = {}
    for m in all_metrics:
        present = sorted(
            ((i, s[m]) for i, s in surprisal_by_i.items() if m in s),
            key=lambda x: -x[1],
        )
        for rank, (i, _) in enumerate(present, start=1):
            cur = best_rank_by_i.get(i)
            if cur is None or rank < cur:
                best_rank_by_i[i] = rank

    n_rankable = len(best_rank_by_i)
    n_keep = max(1, math.ceil(fraction * n_rankable))
    ordered = sorted(best_rank_by_i.items(), key=lambda x: x[1])
    return {i for i, _ in ordered[:n_keep]}


def _filter_by_surprisal(
    records: list[dict[str, str]],
    examples: list[dict[str, Any]],
    fraction: float,
    language: str,
) -> tuple[list[dict[str, str]], Counter[str]]:
    """Apply top-surprisal filtering, grouped by prompt type.

    `records[i]` is the SFT record built from `examples[i]`. Returns the filtered
    records and a Counter of kept-by-prompt-type counts.
    """
    by_kind: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        by_kind[str(ex.get("prompt", "unknown"))].append(i)

    kept: list[int] = []
    kept_counts: Counter[str] = Counter()
    for kind, idxs in by_kind.items():
        programs = [_reconstruct_program(examples[i], language) for i in idxs]
        keep_local = _select_top_surprisal_indices(programs, fraction, language)
        if keep_local is None:
            # No rankable examples in this group: pass them all through.
            print(f"  [top-surprisal] {kind}: no rankable examples, "
                  f"keeping all {len(idxs)}")
            kept.extend(idxs)
            kept_counts[kind] += len(idxs)
            continue
        rankable = sum(1 for p in programs if p is not None)
        for local_i, ex_i in enumerate(idxs):
            if local_i in keep_local:
                kept.append(ex_i)
                kept_counts[kind] += 1
        print(f"  [top-surprisal] {kind}: kept {len(keep_local)}/{rankable} "
              f"rankable ({len(idxs)} total)")

    kept.sort()
    return [records[i] for i in kept], kept_counts


def build_sft_records(
    pickle_paths: list[str | Path],
    success_only: bool,
    outcome_success_values: tuple[str, ...] = ("success",),
    language: str = "dafny",
    top_surprisal_fraction: float | None = None,
) -> tuple[list[dict[str, str]], Counter[str]]:
    """Build TRL/HF records for chat-style SFT.

    Output schema:
      {"messages": <list[dict(role, content)]>, "completion": <str>}

    Returns:
      A tuple of (records, counts_by_prompt_type).

    We reconstruct system+user messages from stored arguments using the language backend's prompt builder.

    For "idea" examples, success is determined by whether the idea led to a
    successful implementation (i.e., there exists an "implement" example with
    a successful outcome that used this idea text).
    """
    from language import Language
    from patch import TEXT_BEFORE_EXAMPLE, TEXT_DIFF_EXAMPLE, TEXT_AFTER_EXAMPLE
    from tasks import discover_tasks

    _pb = Language[language.upper()].get_backend().prompt_builder

    # Task-dispatch fallback for prompt types not handled by the language
    # prompt builder (e.g. lemma_synth, which owns its own message format).
    task_instances: dict[str, Any] = {}
    for _name, cls in discover_tasks().items():
        pt = getattr(cls, 'prompt_type', None)
        if pt and pt not in task_instances:
            task_instances[pt] = cls()

    def reconstruct_chat_messages(kind, args, example_before, example_diff, example_after):
        if kind == "implement":
            return _pb.implement(idea=args.get("idea", ""))
        elif kind == "repair":
            return _pb.repair(program=args.get("program", ""), notes=args.get("notes", ""),
                              example_before=example_before, example_diff=example_diff,
                              example_after=example_after)
        elif kind == "extend":
            return _pb.extend(program=args.get("program", ""),
                              example_before=example_before, example_diff=example_diff,
                              example_after=example_after)
        elif kind == "idea":
            return _pb.idea(repo=args.get("repo", ""), readme=args.get("readme", ""))
        elif kind == "initiate":
            return _pb.initiate(repo=args.get("repo", ""), readme=args.get("readme", ""))
        else:
            return []

    # First pass: collect idea texts that led to successful implementations.
    # An idea is successful if there's an implement example with successful outcome
    # where arguments["idea"] matches the idea's response text.
    successful_idea_texts: set[str] = set()
    for pickle_path in pickle_paths:
        for ex in _iter_distill_examples_from_pickle(pickle_path):
            if ex.get("prompt") != "implement":
                continue
            outcome = ex.get("outcome")
            outcome_s = "" if outcome is None else str(outcome).lower()
            if outcome_s not in outcome_success_values:
                continue
            # This implementation was successful - record its idea text
            args = ex.get("arguments")
            if isinstance(args, dict):
                idea_text = args.get("idea")
                if idea_text:
                    successful_idea_texts.add(str(idea_text))

    # Second pass: build SFT records
    records: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    # Parallel list of source examples, only populated when we will filter by
    # surprisal afterwards (so we can rank within each prompt type).
    source_examples: list[dict[str, Any]] | None = (
        [] if top_surprisal_fraction is not None else None
    )
    for pickle_path in pickle_paths:
        for ex in _iter_distill_examples_from_pickle(pickle_path):
            kind = str(ex.get("prompt", "unknown"))
            args = ex.get("arguments")
            if not isinstance(args, dict):
                continue

            response = ex.get("response")
            if response is None:
                continue
            response_s = str(response)

            # Apply success filtering
            if success_only:
                if kind == "idea":
                    # For ideas, check if the idea led to a successful implementation
                    if response_s not in successful_idea_texts:
                        continue
                else:
                    # For other prompts, use the outcome field
                    outcome = ex.get("outcome")
                    outcome_s = "" if outcome is None else str(outcome).lower()
                    if outcome_s not in outcome_success_values:
                        continue

            messages = reconstruct_chat_messages(
                kind,
                args,
                example_before=TEXT_BEFORE_EXAMPLE,
                example_diff=TEXT_DIFF_EXAMPLE,
                example_after=TEXT_AFTER_EXAMPLE,
            )

            # Fall back to task dispatch for language-specific prompts.
            if not messages:
                task = task_instances.get(kind)
                if task is not None:
                    rec = task.to_training_record(ex)
                    if rec:
                        messages = rec["messages"]

            if not messages:
                continue

            records.append({"prompt": messages, "completion": [{"role": "assistant", "content": response_s}]})
            counts[kind] += 1
            if source_examples is not None:
                source_examples.append(ex)

    if top_surprisal_fraction is not None and source_examples is not None:
        if not (0.0 < top_surprisal_fraction <= 1.0):
            raise ValueError(
                f"top_surprisal_fraction must be in (0, 1], got {top_surprisal_fraction}"
            )
        print(f"\n[top-surprisal] filtering with fraction={top_surprisal_fraction}")
        records, counts = _filter_by_surprisal(
            records, source_examples, top_surprisal_fraction, language
        )

    return records, counts


def build_sft_records_from_tasks(
    pickle_paths: list[str | Path],
    success_only: bool,
    outcome_success_values: tuple[str, ...] = ("success",),
) -> tuple[list[dict[str, str]], Counter[str]]:
    """Build SFT records using task-specific to_training_record() methods.

    This dispatches each example to the appropriate EvaluationTask subclass
    for training record generation, allowing task-specific prompt formatting.

    Falls back to build_sft_records() for unknown prompt types.
    """
    from tasks import discover_tasks

    task_classes = discover_tasks()
    # Map prompt_type -> task instance
    task_instances: dict[str, Any] = {}
    for _name, cls in task_classes.items():
        pt = getattr(cls, 'prompt_type', None)
        if pt and pt not in task_instances:
            task_instances[pt] = cls()

    records: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    for pickle_path in pickle_paths:
        for ex in _iter_distill_examples_from_pickle(pickle_path):
            kind = str(ex.get("prompt", "unknown"))
            if not ex.get("arguments"):
                continue
            response = ex.get("response")
            if response is None:
                continue

            if success_only:
                outcome = ex.get("outcome")
                outcome_s = "" if outcome is None else str(outcome).lower()
                if outcome_s not in outcome_success_values:
                    continue

            task = task_instances.get(kind)
            if task:
                record = task.to_training_record(ex)
                if record:
                    messages = record["messages"]
                    if isinstance(messages, list):
                        messages = json.dumps(messages, ensure_ascii=False)
                    records.append({"messages": messages, "completion": record["completion"]})
                    counts[kind] += 1
            # else: skip unknown prompt types (or could fall back to build_sft_records)

    return records, counts


def _train_with_trl(
    records: list[dict[str, str]],
    model_id: str,
    output_dir: str,
    max_seq_length: int,
    max_steps: int,
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
    max_grad_norm: float = 1.0,
) -> None:
    """Train an SFT model with TRL SFTTrainer + PEFT LoRA.

    Notes:
    - Multi-GPU is handled by Accelerate/Trainer when launched via `accelerate launch`.
    - We keep the implementation conservative: no custom packing tricks, and we train on a
      prompt+completion schema.
    """
    import os

    checkpoint_exists = any(p.startswith('checkpoint-')
                            for p in ((os.path.exists(output_dir) and os.listdir(output_dir)) or []))
    print(f'Checkpoint in {output_dir} exists?', checkpoint_exists)

    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", wandb_project or "formal-disco")
        if wandb_run_name:
            os.environ.setdefault("WANDB_NAME", wandb_run_name)
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        os.environ.setdefault("WANDB_WATCH", "false")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Drop records whose prompt+completion don't fit in max_seq_length.
    # (TRL would otherwise right-truncate and zero out the loss.)
    def _fits(record) -> bool:
        ids = tokenizer.apply_chat_template(
            list(record["prompt"]) + list(record["completion"]),
            tokenize=True, add_generation_prompt=False,
        )
        # apply_chat_template returns a BatchEncoding (dict) for some fast
        # tokenizers and a flat list for others; normalize to the token list.
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return len(ids) <= max_seq_length

    n_before = len(records)
    records = [r for r in records if _fits(r)]
    print(f"[filter] kept {len(records)}/{n_before} records with prompt+completion <= {max_seq_length}")
    if not records:
        raise ValueError("No records to train on (after filtering).")
    ds = Dataset.from_list(records)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="kernels-community/flash-attn2",
        device_map="auto"
    )

    peft_config = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = get_peft_model(model, peft_config)

    report_to = ["wandb"] if use_wandb else ["none"]

    training_args = SFTConfig(
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
        packing=True,
        max_length=int(max_seq_length),
        max_grad_norm=float(max_grad_norm),
    )

    # Treat max_steps as an upper bound: only force it when training for the full
    # num_train_epochs would exceed it. Estimate per-epoch steps from the record
    # count (ignoring packing): with the prompt/completion schema TRL doesn't
    # actually compact multiple records into one sequence, so a tokens/max_len
    # estimate would massively underestimate the real step count.
    world_size = 4  # This is hardcoded for now: slurm/sft.sbatch requests 4 GPUs
    effective_batch_size = (
        int(per_device_train_batch_size) * int(gradient_accumulation_steps) * world_size
    )
    steps_per_epoch = max(1, math.ceil(len(records) / effective_batch_size))
    estimated_epoch_steps = math.ceil(steps_per_epoch * float(num_train_epochs))

    if max_steps and int(max_steps) > 0 and estimated_epoch_steps > int(max_steps):
        training_args.max_steps = int(max_steps)
        print(
            f"[steps] capping at max_steps={max_steps} "
            f"(full {num_train_epochs} epochs ≈ {estimated_epoch_steps} steps; "
            f"per-epoch ≈ {steps_per_epoch}, effective_batch={effective_batch_size})"
        )
    else:
        print(
            f"[steps] running num_train_epochs={num_train_epochs} "
            f"(≈ {estimated_epoch_steps} steps; per-epoch ≈ {steps_per_epoch}, "
            f"effective_batch={effective_batch_size}); max_steps cap = {max_steps or 'unset'}"
        )

    # Temporary: log per-step loss to <output_dir>/train_loss.json for offline inspection.
    from transformers import TrainerCallback

    class JSONLossLogger(TrainerCallback):
        def __init__(self, path: str):
            self.path = path
            self.history: list[dict[str, float]] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            entry = {"step": int(state.global_step)}
            for k in ("loss", "learning_rate", "grad_norm", "epoch"):
                if k in logs:
                    try:
                        entry[k] = float(logs[k])
                    except Exception:
                        pass
            self.history.append(entry)
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.history, f, indent=2)

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=training_args,
        callbacks=[JSONLossLogger(os.path.join(output_dir, "train_loss.json"))],
    )

    trainer.train(
        # Resume from checkpoint if any checkpoint directory exists.
        # Setting this to True when there's no checkpoint raises an exception,
        # so we need to check first.
        # Also note that, confusingly, SFTConfig has a resume_from_checkpoint property
        # that train() ignores since it has its own argument too.
        resume_from_checkpoint=checkpoint_exists,
    )

    # Save adapter + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Merge LoRA into base model weights and save a merged checkpoint for inference.
    # vLLM can only load this merged model, not the model with LoRA adapters.
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
        # One or more agenda checkpoint pickle paths.
        # Hydra config (`config/distill.yaml`) may provide this as a single string
        # or a YAML list of strings.
        data: str | list[str] = "local-agenda.pkl"
        model_id: str = DEFAULT_HF_MODEL_ID
        output_dir: str = "sft-out"
        # Formal language (dafny, verus)
        language: str = "dafny"
        success_only: bool = True
        # Treat GOAL_UNPROVEN as success? (optional)
        treat_goal_unproven_as_success: bool = False
        # If set (in (0, 1]), filter SFT examples to the top fraction by best
        # per-metric surprisal rank within each prompt type. Used as an
        # entropy-maximization-style data selection for iterative SFT.
        top_surprisal_fraction: Optional[float] = None

        # Training
        max_steps: Optional[int] = None
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
        max_grad_norm: float = 1.0

        lora_r: int = 16
        lora_alpha: int = 32
        lora_dropout: float = 0.05
        lora_target_modules: list[str] | None = None

        wandb: bool = True
        wandb_project: str | None = None
        wandb_run_name: str | None = None

        dry_run: bool = False

    # Collect Hydra-style overrides from sys.argv (everything after "sft" was stripped).
    overrides = [a for a in sys.argv[1:] if "=" in a]

    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="distill", overrides=overrides)

    # Merge user cfg into typed defaults
    c = SftCfg(**{k: v for k, v in cfg.items()})

    success_values = ("success",)
    if c.treat_goal_unproven_as_success:
        success_values = ("success", "goal_unproven")

    pickle_paths: list[str]
    if isinstance(c.data, str):
        pickle_paths = [c.data]
    else:
        pickle_paths = [str(p) for p in c.data]

    print('Input data files:', c.data)

    records, counts = build_sft_records(
        pickle_paths=pickle_paths,
        success_only=bool(c.success_only),
        outcome_success_values=success_values,
        language=c.language,
        top_surprisal_fraction=c.top_surprisal_fraction,
    )

    # Print training data statistics
    print("\n=== SFT Training Data ===")
    print(f"Total examples: {len(records)}")
    print("By prompt type:")
    for prompt_type, count in sorted(counts.items()):
        print(f"  {prompt_type}: {count}")
    print()

    if c.dry_run:
        DRY_RUN_OUT = Path('distil.dry-run.json').absolute()
        print(f"Dry run: writing training data to {DRY_RUN_OUT} and exiting...")

        with open(DRY_RUN_OUT, 'w') as f:
            json.dump(records, f, indent=4)

        return

    out_dir = os.path.abspath(c.output_dir)

    _train_with_trl(
        records=records,
        model_id=c.model_id or DEFAULT_HF_MODEL_ID,
        output_dir=out_dir,
        max_seq_length=c.max_seq_length,
        max_steps=c.max_steps,
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
        max_grad_norm=c.max_grad_norm,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="distill.py")
    sub = parser.add_subparsers(dest="command", required=True)

    stats_p = sub.add_parser("stats", help="Summarize distillation objects in an agenda pickle")
    stats_p.add_argument("-d", "--data", required=True, help="Path to agenda checkpoint pickle")
    stats_p.set_defaults(func=_cmd_stats)

    # `sft` is Hydra-based.
    sft_p = sub.add_parser("sft", help="Train an SFT model from distil examples (Hydra config)")
    sft_p.set_defaults(func=lambda _args: _main_sft())

    ns, _ = parser.parse_known_args(argv)

    # Remove command from argv (hack to make Hydra not think sft is an override).
    sys.argv = [s for s in sys.argv if s not in ("stats", "sft")]

    ns.func(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
