#!/usr/bin/env python3
"""Compare multiple agenda runs: success rates and diversity/complexity metrics."""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from language import Language, Program

PLOTS_CONFIG_PATH = Path(__file__).parent / "plots.config.json"
PLOTS_DIR = Path(__file__).parent / "plots"


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


def _detect_language(agenda: dict) -> Language:
    """Infer the language from object types in the agenda (e.g. 'verus-program' -> VERUS)."""
    for o in agenda.get("objects", {}).values():
        if hasattr(o, "type") and o.type.endswith("-program"):
            lang_name = o.type.removesuffix("-program").upper()
            try:
                return Language[lang_name]
            except KeyError as exc:
                raise ValueError(f"Unknown language in agenda object type: {o.type}") from exc
    raise ValueError("Could not detect language from agenda objects")


def diversity_complexity_table(agendas: dict[str, dict]) -> None:
    print("=" * 80)
    print("DIVERSITY & COMPLEXITY (dataset/ programs only)")
    print("=" * 80)

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

        lang = _detect_language(a)
        backend = lang.get_backend()

        all_complexity = []
        all_features = []
        n_parseable = 0

        for k, o in dataset_objs.items():
            text = o.content.decode("utf-8") if isinstance(o.content, bytes) else o.content
            prog = Program(text, lang)
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

    # Collect complexity/feature keys from all agendas (language-dependent)
    complexity_keys = sorted({
        ck for label in labels
        if (d := per_agenda[label]) and d["complexity"]
        for ck in d["complexity"][0]
    })
    feature_keys = sorted({
        fk for label in labels
        if (d := per_agenda[label]) and d["features"]
        for fk in d["features"][0]
    })

    # Complexity metrics: aggregate across all programs
    rows.append(("", [""] * len(labels)))
    rows.append(("COMPLEXITY (mean of per-program means)", [""] * len(labels)))

    for ck in complexity_keys:
        row_vals = []
        for label in labels:
            d = per_agenda[label]
            if not d or not d["complexity"]:
                row_vals.append("-")
                continue
            per_prog_means = [_safe_mean(cx[ck]) for cx in d["complexity"] if ck in cx]
            row_vals.append(f"{_safe_mean(per_prog_means):.2f}")
        rows.append((f"  {ck}", row_vals))

    # Diversity metrics: entropy of pooled feature counters
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


def load_dafnybench(directory: str) -> dict:
    """Load .dfy files from a directory into a fake agenda dict for metrics."""
    from types import SimpleNamespace

    dfy_dir = Path(directory)
    objects = {}
    for dfy_file in sorted(dfy_dir.glob("*.dfy")):
        key = f"dataset/{dfy_file.stem}"
        objects[key] = SimpleNamespace(
            content=dfy_file.read_text(),
            type="dafny-program",
            properties={},
        )
    return {"objects": objects, "tasks": {}, "status": {}}


def load_plots_config() -> dict:
    if PLOTS_CONFIG_PATH.exists():
        with open(PLOTS_CONFIG_PATH) as f:
            return json.load(f)
    return {}


def get_run_name(config: dict, path: str) -> str:
    """Get user-readable name for an agenda file, falling back to run_label."""
    name_map = config.get("run_name", {})
    basename = Path(path).name
    if basename in name_map:
        return name_map[basename]
    stem = Path(path).stem
    if stem in name_map:
        return name_map[stem]
    return run_label(path)


def save_chart(chart, name: str) -> None:
    """Save an Altair chart as both .svg and .png in the plots directory."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = PLOTS_DIR / f"{name}.svg"
    png_path = PLOTS_DIR / f"{name}.png"
    chart.save(str(svg_path), format="svg")
    chart.save(str(png_path), format="png", scale_factor=2)
    print(f"Saved {svg_path}")
    print(f"Saved {png_path}")


# ---------------------------------------------------------------------------
# Plot: Task Success Rates
# ---------------------------------------------------------------------------

def plot_task_success_rates(agenda_paths: list[str]) -> None:
    import altair as alt

    config = load_plots_config()
    rows = []

    for path in agenda_paths:
        label = get_run_name(config, path)
        agenda = load_agenda(path)
        task_outcomes = agenda.get("task_outcomes", {})

        for task_type, outcomes in task_outcomes.items():
            done = outcomes.get("DONE", 0)
            failed = outcomes.get("FAILED", 0)
            attempted = outcomes.get("ATTEMPTED", 0)
            total = done + failed + attempted
            if total == 0:
                continue
            rate = done / total
            # 95% CI using Wilson score interval
            z = 1.96
            denom = 1 + z**2 / total
            center = (rate + z**2 / (2 * total)) / denom
            margin = z * np.sqrt((rate * (1 - rate) + z**2 / (4 * total)) / total) / denom
            ci_lo = max(0, center - margin)
            ci_hi = min(1, center + margin)
            rows.append({
                "Task Type": task_type,
                "Run": label,
                "Success Rate": rate,
                "CI Low": ci_lo,
                "CI High": ci_hi,
                "n": total,
            })

    if not rows:
        print("No task outcome data found in the provided agendas.")
        return

    df = alt.Data(values=rows)

    base = alt.Chart(df).properties(width=120)

    bars = base.mark_bar().encode(
        x=alt.X("Run:N", axis=None),
        y=alt.Y("Success Rate:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Run:N", title="Run"),
    )

    error_bars = base.mark_errorbar(color="black").encode(
        x=alt.X("Run:N", axis=None),
        y=alt.Y("CI Low:Q", title="Success Rate"),
        y2=alt.Y2("CI High:Q"),
    )

    chart = (bars + error_bars).facet(
        column=alt.Column("Task Type:N", title="Task Type"),
        spacing=10,
    ).properties(
        title="Task Success Rates by Run",
    )

    save_chart(chart, "task-success-rate")


# ---------------------------------------------------------------------------
# Helpers: extract verified programs (longest per source idea)
# ---------------------------------------------------------------------------

def _extract_verified_programs(agenda: dict, language: Language) -> list[Program]:
    """Extract verified programs from an agenda, keeping the longest per source idea.

    Each dataset/ object traces back to a programs/ parent (the original init-prog).
    When multiple dataset entries share the same root parent, keep the longest.
    Only includes programs with verification_outcome == 'SUCCESS'.
    """
    objs = agenda["objects"]

    # Group dataset/ objects by their root parent (the init-prog)
    by_root: dict[str, list] = {}
    for key, obj in objs.items():
        if not key.startswith("dataset/"):
            continue
        vo = obj.properties.get("verification_outcome", obj.properties.get("verification_status", ""))
        if str(vo).upper() != "SUCCESS" and vo != "success":
            continue
        root = obj.parents[0] if obj.parents else key
        text = obj.content.decode("utf-8") if isinstance(obj.content, bytes) else obj.content
        by_root.setdefault(root, []).append(text)

    # Keep the longest per root
    programs = []
    for root, texts in by_root.items():
        longest = max(texts, key=len)
        programs.append(Program(longest, language))
    return programs


# ---------------------------------------------------------------------------
# Plot: Program Complexity
# ---------------------------------------------------------------------------

def plot_program_complexity(agenda_paths: list[str], language: Language) -> None:
    import altair as alt

    config = load_plots_config()
    backend = language.get_backend()
    rows = []

    for path in agenda_paths:
        label = get_run_name(config, path)
        agenda = load_agenda(path)
        programs = _extract_verified_programs(agenda, language)
        print(f"  {label}: {len(programs)} verified programs")

        for prog in programs:
            try:
                cx = backend.complexity(prog)
            except Exception:
                continue
            for metric, values in cx.items():
                if not values:
                    continue
                rows.append({
                    "Run": label,
                    "Metric": metric,
                    "Value": float(np.mean(values)),
                })

    if not rows:
        print("No complexity data found.")
        return

    df = alt.Data(values=rows)

    chart = alt.Chart(df).mark_boxplot(extent="min-max").encode(
        x=alt.X("Run:N", axis=None),
        y=alt.Y("Value:Q"),
        color=alt.Color("Run:N", title="Run"),
    ).properties(
        width=120,
    ).facet(
        column=alt.Column("Metric:N", title="Complexity Metric"),
        spacing=10,
    ).resolve_scale(
        y="independent",
    ).properties(
        title="Program Complexity (per-program mean, verified programs)",
    )

    save_chart(chart, "program-complexity")


# ---------------------------------------------------------------------------
# Plot: Program Diversity
# ---------------------------------------------------------------------------

def plot_program_diversity(agenda_paths: list[str], language: Language) -> None:
    import altair as alt

    config = load_plots_config()
    backend = language.get_backend()

    # For diversity we compute corpus-level entropy and unique-count per feature type.
    # We'll show two sub-plots: entropy (bits) and unique feature count.
    entropy_rows = []
    unique_rows = []

    for path in agenda_paths:
        label = get_run_name(config, path)
        agenda = load_agenda(path)
        programs = _extract_verified_programs(agenda, language)
        print(f"  {label}: {len(programs)} verified programs")

        pooled: dict[str, Counter] = {}
        for prog in programs:
            try:
                ft = backend.feature_sets(prog)
            except Exception:
                continue
            for metric, counter in ft.items():
                pooled.setdefault(metric, Counter()).update(counter)

        for metric, counter in pooled.items():
            entropy_rows.append({
                "Run": label,
                "Feature": metric,
                "Entropy (bits)": _entropy(counter),
            })
            unique_rows.append({
                "Run": label,
                "Feature": metric,
                "Unique Features": len(counter),
            })

    if not entropy_rows:
        print("No diversity data found.")
        return

    # Entropy chart
    df_ent = alt.Data(values=entropy_rows)
    chart_ent = alt.Chart(df_ent).mark_bar().encode(
        x=alt.X("Run:N", axis=None),
        y=alt.Y("Entropy (bits):Q"),
        color=alt.Color("Run:N", title="Run"),
    ).properties(width=120).facet(
        column=alt.Column("Feature:N", title="Feature"),
        spacing=10,
    ).resolve_scale(y="independent").properties(
        title="Program Diversity: Entropy (verified programs, longest per idea)",
    )
    save_chart(chart_ent, "program-diversity-entropy")

    # Unique count chart
    df_uniq = alt.Data(values=unique_rows)
    chart_uniq = alt.Chart(df_uniq).mark_bar().encode(
        x=alt.X("Run:N", axis=None),
        y=alt.Y("Unique Features:Q"),
        color=alt.Color("Run:N", title="Run"),
    ).properties(width=120).facet(
        column=alt.Column("Feature:N", title="Feature"),
        spacing=10,
    ).resolve_scale(y="independent").properties(
        title="Program Diversity: Unique Features (verified programs, longest per idea)",
    )
    save_chart(chart_uniq, "program-diversity-unique")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare agenda runs.")
    subparsers = parser.add_subparsers(dest="command")

    # tables (default / legacy mode)
    p_tables = subparsers.add_parser("tables", help="Print comparison tables (default)")
    p_tables.add_argument("agendas", nargs="*", help="Paths to agenda .pkl files")
    p_tables.add_argument(
        "--dafnybench",
        metavar="DIR",
        default=None,
        help="Path to DafnyBench ground_truth directory",
    )

    # plot-task-success-rates
    p_tsr = subparsers.add_parser("plot-task-success-rates",
                                  help="Bar chart of success rates per task type")
    p_tsr.add_argument("agendas", nargs="+", help="Paths to agenda .pkl files")

    # plot-program-complexity
    p_cx = subparsers.add_parser("plot-program-complexity",
                                 help="Box plot of complexity metrics per run")
    p_cx.add_argument("agendas", nargs="+", help="Paths to agenda .pkl files")
    p_cx.add_argument("--language", default="dafny", help="Language (default: dafny)")

    # plot-program-diversity
    p_div = subparsers.add_parser("plot-program-diversity",
                                  help="Bar charts of diversity metrics per run")
    p_div.add_argument("agendas", nargs="+", help="Paths to agenda .pkl files")
    p_div.add_argument("--language", default="dafny", help="Language (default: dafny)")

    args = parser.parse_args()

    # Default to "tables" when no subcommand given (legacy behavior)
    if args.command is None:
        args = p_tables.parse_args(sys.argv[1:])
        args.command = "tables"

    if args.command == "plot-task-success-rates":
        plot_task_success_rates(args.agendas)
        return

    if args.command in ("plot-program-complexity", "plot-program-diversity"):
        lang = Language[args.language.upper()]
        if args.command == "plot-program-complexity":
            plot_program_complexity(args.agendas, lang)
        else:
            plot_program_diversity(args.agendas, lang)
        return

    # tables mode
    if not args.agendas and not getattr(args, "dafnybench", None):
        parser.error("Provide at least one agenda .pkl file or --dafnybench DIR")

    agendas: dict[str, dict] = {}

    if getattr(args, "dafnybench", None):
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
