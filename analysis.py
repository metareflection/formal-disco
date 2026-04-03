#!/usr/bin/env python3
"""Compare multiple agenda runs: success rates and diversity/complexity metrics."""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from language import Language, Program


def load_agenda(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def run_label(path: str) -> str:
    """Derive a short label from the agenda filename."""
    s = Path(path).stem
    for prefix in ["agenda-sn-monolithic-", "agenda-sn-monolithic__", "agenda-"]:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


# ---------------------------------------------------------------------------
# Table 0: Budget (total attempts per agenda and per task type)
# ---------------------------------------------------------------------------

def budget_table(agendas: dict[str, dict]) -> None:
    print("=" * 80)
    print("BUDGET (task attempts)")
    print("=" * 80)

    labels = list(agendas.keys())
    col_w = max(20, *(len(l) for l in labels)) + 2
    header = f"{'Metric':<30}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))

    rows: list[tuple[str, list[str]]] = []

    # Total attempts
    row_vals = []
    for label, a in agendas.items():
        row_vals.append(str(a.get("total_attempts", "?")))
    rows.append(("Total attempts", row_vals))

    # Per task type from task_outcomes
    all_types = sorted({
        tt for a in agendas.values() for tt in a.get("task_outcomes", {})
    })
    for tt in all_types:
        row_vals = []
        for label, a in agendas.items():
            outcomes = a.get("task_outcomes", {}).get(tt, {})
            row_vals.append(str(sum(outcomes.values())))
        rows.append((f"  {tt}", row_vals))

    for metric, vals in rows:
        line = f"{metric:<30}" + "".join(f"{v:>{col_w}}" for v in vals)
        print(line)
    print()


# ---------------------------------------------------------------------------
# Table 1: Success rates
# ---------------------------------------------------------------------------

def success_rate_table(agendas: dict[str, dict]) -> None:
    print("=" * 80)
    print("SUCCESS RATES")
    print("=" * 80)

    # Header
    labels = list(agendas.keys())
    col_w = max(20, *(len(l) for l in labels)) + 2
    header = f"{'Metric':<30}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))

    rows: list[tuple[str, list[str]]] = []

    for label, a in agendas.items():
        tasks = a["tasks"]
        statuses = a["status"]
        total_attempts = a.get("total_attempts", "?")

        # Per task-type breakdown
        by_type: dict[str, Counter] = {}
        for tid, t in tasks.items():
            s = statuses.get(tid)
            if s is None:
                continue
            by_type.setdefault(t.type, Counter())[s.work_status] += 1

        agendas[label]["_by_type"] = by_type
        agendas[label]["_total_attempts"] = total_attempts

    # Collect all task types
    all_types = sorted({
        tt for a in agendas.values() for tt in a.get("_by_type", {})
    })

    # Total attempts
    row_vals = []
    for label, a in agendas.items():
        row_vals.append(str(a["_total_attempts"]))
    rows.append(("Total attempts", row_vals))

    for tt in all_types:
        # Total tasks of this type
        row_vals = []
        for label, a in agendas.items():
            c = a["_by_type"].get(tt, Counter())
            row_vals.append(str(sum(c.values())))
        rows.append((f"  [{tt}] total tasks", row_vals))

        # DONE / FAILED / other
        for ws in ["DONE", "FAILED", "NEW", "ATTEMPTED", "DOING"]:
            any_nonzero = any(
                a["_by_type"].get(tt, Counter()).get(ws, 0) > 0
                for a in agendas.values()
            )
            if not any_nonzero:
                continue
            row_vals = []
            for label, a in agendas.items():
                c = a["_by_type"].get(tt, Counter())
                total = sum(c.values())
                n = c.get(ws, 0)
                pct = f" ({100 * n / total:.1f}%)" if total > 0 else ""
                row_vals.append(f"{n}{pct}")
            rows.append((f"    {ws}", row_vals))

    # Verification outcome on program objects
    rows.append(("", [""] * len(labels)))
    rows.append(("Program verification outcomes", [""] * len(labels)))

    for label, a in agendas.items():
        programs = {k: o for k, o in a["objects"].items() if k.startswith("programs/")}
        v_counts = Counter(o.properties.get("verification_outcome") for o in programs.values())
        a["_v_counts"] = v_counts
        a["_n_programs"] = len(programs)

    for outcome in ["SUCCESS", "GOAL_UNPROVEN", "FAIL"]:
        row_vals = []
        for label, a in agendas.items():
            n = a["_v_counts"].get(outcome, 0)
            total = a["_n_programs"]
            pct = f" ({100 * n / total:.1f}%)" if total > 0 else ""
            row_vals.append(f"{n}{pct}")
        rows.append((f"  {outcome}", row_vals))

    # Verification history patterns (how repairs went)
    rows.append(("", [""] * len(labels)))
    rows.append(("Repair outcomes", [""] * len(labels)))

    for label, a in agendas.items():
        statuses = a["status"]
        vh_patterns: Counter[str] = Counter()
        for s in statuses.values():
            vh = tuple(s.worker_notes.get("verification", []))
            if not vh:
                continue
            # Classify: succeeded on first try, succeeded after repair, all failed
            if vh[-1] == "SUCCESS":
                vh_patterns["repaired"] += 1
            else:
                vh_patterns["all_failed"] += 1
        a["_vh_patterns"] = vh_patterns

    # Tasks that succeeded on first try (repair_attempts == 0 or verification == [SUCCESS])
    row_vals = []
    for label, a in agendas.items():
        statuses = a["status"]
        n = sum(
            1 for s in statuses.values()
            if s.worker_notes.get("repair_attempts", -1) == 0
            and s.worker_notes.get("verification") == []
        )
        # Actually: first-try success means the initial generation succeeded,
        # which means repair_attempts == 0 and status is DONE
        n_first = sum(
            1 for s in statuses.values()
            if s.worker_notes.get("repair_attempts", -1) == 0
            and str(s.work_status) == "DONE"
        )
        row_vals.append(str(n_first))
    rows.append(("  Succeeded (no repair)", row_vals))

    row_vals = []
    for label, a in agendas.items():
        row_vals.append(str(a["_vh_patterns"].get("repaired", 0)))
    rows.append(("  Succeeded (after repair)", row_vals))

    row_vals = []
    for label, a in agendas.items():
        row_vals.append(str(a["_vh_patterns"].get("all_failed", 0)))
    rows.append(("  Failed (all repairs failed)", row_vals))

    # Dataset size
    rows.append(("", [""] * len(labels)))
    row_vals = []
    for label, a in agendas.items():
        n = sum(1 for k in a["objects"] if k.startswith("dataset/"))
        row_vals.append(str(n))
    rows.append(("Dataset programs (verified)", row_vals))

    # Print
    for metric, vals in rows:
        line = f"{metric:<30}" + "".join(f"{v:>{col_w}}" for v in vals)
        print(line)
    print()


# ---------------------------------------------------------------------------
# Table 2: Diversity & Complexity
# ---------------------------------------------------------------------------

def _safe_mean(xs):
    return float(np.mean(xs)) if xs else 0.0


def _safe_median(xs):
    return float(np.median(xs)) if xs else 0.0


def _entropy(counter: Counter) -> float:
    """Shannon entropy in bits."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counter.values()])
    probs = probs[probs > 0]
    return max(0.0, -float(np.sum(probs * np.log2(probs))))


def diversity_complexity_table(agendas: dict[str, dict]) -> None:
    print("=" * 80)
    print("DIVERSITY & COMPLEXITY (dataset/ programs only)")
    print("=" * 80)

    backend = Language.DAFNY.get_backend()
    labels = list(agendas.keys())
    col_w = max(20, *(len(l) for l in labels)) + 2

    # Compute metrics for each agenda
    per_agenda: dict[str, dict] = {}
    for label, a in agendas.items():
        dataset_objs = {
            k: o for k, o in a["objects"].items() if k.startswith("dataset/")
        }
        if not dataset_objs:
            per_agenda[label] = None
            continue

        all_complexity = []
        all_features = []
        n_parseable = 0

        for k, o in dataset_objs.items():
            text = o.content.decode("utf-8") if isinstance(o.content, bytes) else o.content
            prog = Program(text, Language.DAFNY)
            try:
                cx = backend.complexity(prog)
                ft = backend.feature_sets(prog)
                all_complexity.append(cx)
                all_features.append(ft)
                n_parseable += 1
            except Exception:
                continue

        per_agenda[label] = {
            "n_total": len(dataset_objs),
            "n_parseable": n_parseable,
            "complexity": all_complexity,
            "features": all_features,
        }

    # Print header
    header = f"{'Metric':<40}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))

    rows: list[tuple[str, list[str]]] = []

    row_vals = []
    for label in labels:
        d = per_agenda[label]
        row_vals.append(str(d["n_total"]) if d else "0")
    rows.append(("Programs in dataset/", row_vals))

    # Complexity metrics: aggregate across all programs
    complexity_keys = ["body_sizes", "n_loops_per_method", "n_idents_in_asserts", "n_idents_in_invs"]
    rows.append(("", [""] * len(labels)))
    rows.append(("COMPLEXITY (mean of per-program means)", [""] * len(labels)))

    for ck in complexity_keys:
        row_vals = []
        for label in labels:
            d = per_agenda[label]
            if not d or not d["complexity"]:
                row_vals.append("-")
                continue
            per_prog_means = [_safe_mean(cx[ck]) for cx in d["complexity"]]
            row_vals.append(f"{_safe_mean(per_prog_means):.2f}")
        rows.append((f"  {ck}", row_vals))

    # Diversity metrics: entropy of pooled feature counters
    feature_keys = [
        "subject_words", "invariant_templates", "assert_templates",
        "ensures_templates", "requires_templates", "loop_skeletons",
    ]
    rows.append(("", [""] * len(labels)))
    rows.append(("DIVERSITY (entropy in bits, pooled)", [""] * len(labels)))

    for fk in feature_keys:
        row_vals = []
        for label in labels:
            d = per_agenda[label]
            if not d or not d["features"]:
                row_vals.append("-")
                continue
            pooled = Counter()
            for ft in d["features"]:
                pooled.update(ft.get(fk, Counter()))
            row_vals.append(f"{_entropy(pooled):.2f}")
        rows.append((f"  {fk}", row_vals))

    # Also: unique feature counts
    rows.append(("", [""] * len(labels)))
    rows.append(("DIVERSITY (unique features, pooled)", [""] * len(labels)))

    for fk in feature_keys:
        row_vals = []
        for label in labels:
            d = per_agenda[label]
            if not d or not d["features"]:
                row_vals.append("-")
                continue
            pooled = Counter()
            for ft in d["features"]:
                pooled.update(ft.get(fk, Counter()))
            row_vals.append(str(len(pooled)))
        rows.append((f"  {fk}", row_vals))

    for metric, vals in rows:
        line = f"{metric:<40}" + "".join(f"{v:>{col_w}}" for v in vals)
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_dafnybench(directory: str) -> dict:
    """Load .dfy files from a directory into a fake agenda dict for metrics."""
    from types import SimpleNamespace

    dfy_dir = Path(directory)
    objects = {}
    for dfy_file in sorted(dfy_dir.glob("*.dfy")):
        key = f"dataset/{dfy_file.stem}"
        objects[key] = SimpleNamespace(
            content=dfy_file.read_text(),
            properties={},
        )
    return {"objects": objects, "tasks": {}, "status": {}}


def main():
    parser = argparse.ArgumentParser(description="Compare agenda runs.")
    parser.add_argument("agendas", nargs="*", help="Paths to agenda .pkl files")
    parser.add_argument(
        "--dafnybench",
        metavar="DIR",
        default=None,
        help="Path to DafnyBench ground_truth directory (e.g. data/DafnyBench/DafnyBench/dataset/ground_truth/)",
    )
    args = parser.parse_args()

    if not args.agendas and not args.dafnybench:
        parser.error("Provide at least one agenda .pkl file or --dafnybench DIR")

    agendas: dict[str, dict] = {}

    if args.dafnybench:
        agendas["DafnyBench"] = load_dafnybench(args.dafnybench)
        n = len([k for k in agendas["DafnyBench"]["objects"] if k.startswith("dataset/")])
        print(f"Loaded {n} programs from DafnyBench ({args.dafnybench})")

    for path in args.agendas:
        label = run_label(path)
        agendas[label] = load_agenda(path)
        print(f"Loaded {path} as '{label}'")
    print()

    if args.agendas:
        budget_table(agendas)
        success_rate_table(agendas)
    diversity_complexity_table(agendas)


if __name__ == "__main__":
    main()
