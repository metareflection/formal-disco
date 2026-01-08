#!/usr/bin/env python3

"""Distillation utilities.

Currently supports:
  python distill.py stats -d <agenda_pickle>

This prints summary statistics for objects written under the `distil/` prefix.
These objects are expected (by convention) to be JSON documents with keys:
  - prompt: str
  - arguments: dict
  - response: str
  - outcome: str

The agenda pickle is produced by `LocalAgenda` checkpointing.
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="distill.py")
    sub = parser.add_subparsers(dest="command", required=True)

    stats_p = sub.add_parser("stats", help="Summarize distillation objects in an agenda pickle")
    stats_p.add_argument("-d", "--data", required=True, help="Path to agenda checkpoint pickle")
    stats_p.set_defaults(func=_cmd_stats)

    ns = parser.parse_args(argv)
    ns.func(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
