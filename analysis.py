#!/usr/bin/env python3
"""Compare multiple agenda runs: success rates and feature-set metrics."""

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
                obj_path = getattr(o, "path", "<path attribute not available>")
                raise ValueError(
                    f"Unknown language in agenda object type: {o.type} (object path: {obj_path})"
                ) from exc
    raise ValueError("Could not detect language from agenda objects")


def _is_numeric_counter(counter: Counter) -> bool:
    """A Counter whose keys are all int (and non-bool) supports median/p90."""
    if not counter:
        return False
    return all(isinstance(k, int) and not isinstance(k, bool) for k in counter.keys())


def diversity_complexity_table(agendas: dict[str, dict]) -> None:
    print("=" * 80)
    print("DIVERSITY (dataset/ programs only)")
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

        all_features = []
        n_parseable = 0

        for k, o in dataset_objs.items():
            text = o.content.decode("utf-8") if isinstance(o.content, bytes) else o.content
            prog = Program(text, lang)
            try:
                ft = backend.feature_sets(prog)
                all_features.append(ft)
                n_parseable += 1
            except Exception:
                continue

        per_agenda[label] = {
            "n_total": len(dataset_objs),
            "n_parseable": n_parseable,
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

    feature_keys = sorted({
        fk for label in labels
        if (d := per_agenda[label]) and d["features"]
        for fk in d["features"][0]
    })

    # Numeric features: per-program mean of the multiset, averaged across programs.
    numeric_keys = [
        fk for fk in feature_keys
        if any((d := per_agenda[label]) and d["features"]
               and _is_numeric_counter(d["features"][0].get(fk, Counter()))
               for label in labels)
    ]
    if numeric_keys:
        rows.append(("", [""] * len(labels)))
        rows.append(("NUMERIC FEATURES (mean of per-program means)", [""] * len(labels)))
        for ck in numeric_keys:
            row_vals = []
            for label in labels:
                d = per_agenda[label]
                if not d or not d["features"]:
                    row_vals.append("-")
                    continue
                per_prog_means = [
                    _safe_mean(list(ft[ck].elements()))
                    for ft in d["features"] if ck in ft and ft[ck]
                ]
                row_vals.append(f"{_safe_mean(per_prog_means):.2f}")
            rows.append((f"  {ck}", row_vals))

    # Diversity: entropy of pooled feature counters.
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

    # Also: unique feature counts.
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


def resolve_labels(agenda_paths: list[str], labels_arg: str | None) -> list[str]:
    """Resolve display labels for agendas: --labels CSV overrides config."""
    if labels_arg:
        labels = [l.strip() for l in labels_arg.split(",")]
        if len(labels) != len(agenda_paths):
            raise ValueError(
                f"--labels has {len(labels)} entries but {len(agenda_paths)} agendas given"
            )
        return labels
    config = load_plots_config()
    return [get_run_name(config, p) for p in agenda_paths]


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

def plot_task_success_rates(
    agenda_paths: list[str],
    labels: list[str] | None = None,
    name: str = "task-success-rate",
) -> None:
    import altair as alt

    if labels is None:
        config = load_plots_config()
        labels = [get_run_name(config, p) for p in agenda_paths]
    rows = []

    for path, label in zip(agenda_paths, labels):
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

    save_chart(chart, name)


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

def plot_program_complexity(
    agenda_paths: list[str],
    language: Language,
    labels: list[str] | None = None,
    name: str = "program-complexity",
    dafnybench_dir: str | None = None,
    absolute: bool = False,
    x_clip_percentile: float | None = None,
    lower_percentile: float = 25.0,
) -> None:
    import altair as alt

    if labels is None:
        config = load_plots_config()
        labels = [get_run_name(config, p) for p in agenda_paths]
    backend = language.get_backend()
    rows = []

    for path, label in zip(agenda_paths, labels):
        agenda = load_agenda(path)
        programs = _extract_verified_programs(agenda, language)
        print(f"  {label}: {len(programs)} verified programs")

        for prog in programs:
            try:
                fs = backend.feature_sets(prog)
            except Exception:
                continue
            for metric, counter in fs.items():
                if not counter or not _is_numeric_counter(counter):
                    continue
                rows.append({
                    "Run": label,
                    "Metric": metric,
                    "Value": float(np.mean(list(counter.elements()))),
                })

    if not rows:
        print("No numeric-feature data found.")
        return

    if absolute and not dafnybench_dir:
        raise ValueError("--absolute requires --dafnybench (for the threshold)")
    if x_clip_percentile is not None and not dafnybench_dir:
        raise ValueError("--x-clip-percentile requires --dafnybench")

    # DafnyBench: load and (a) compute per-metric mean, (b) include as another run.
    ref_means: dict[str, float] = {}
    if dafnybench_dir:
        from language import Program
        from collections import defaultdict
        per_metric: dict[str, list[float]] = defaultdict(list)
        n_files = 0
        for dfy_file in sorted(Path(dafnybench_dir).glob("*.dfy")):
            try:
                prog = Program(dfy_file.read_text(), language)
                fs = backend.feature_sets(prog)
            except Exception:
                continue
            n_files += 1
            for metric, counter in fs.items():
                if not counter or not _is_numeric_counter(counter):
                    continue
                v = float(np.mean(list(counter.elements())))
                per_metric[metric].append(v)
                rows.append({"Run": "DafnyBench", "Metric": metric, "Value": v})
        ref_lower = {m: float(np.percentile(vs, lower_percentile))
                     for m, vs in per_metric.items()}
        labels = list(labels) + ["DafnyBench"]
        print(f"  DafnyBench: {n_files} programs added; "
              f"per-metric p{lower_percentile}: {ref_lower}")
        if x_clip_percentile is not None:
            x_clip = {m: float(np.percentile(vs, x_clip_percentile))
                      for m, vs in per_metric.items()}
            print(f"  x-axis clipped at DafnyBench p{x_clip_percentile}: {x_clip}")
        else:
            x_clip = {}
    else:
        ref_lower = {}
        x_clip = {}

    # In absolute mode: filter to programs above DafnyBench's lower-percentile
    # per metric, so the count violins reflect "non-trivial" volume.
    if absolute:
        n_before = len(rows)
        rows = [r for r in rows if r["Value"] > ref_lower.get(r["Metric"], float("inf"))]
        print(f"  absolute mode: kept {len(rows)}/{n_before} rows "
              f"(above DafnyBench p{lower_percentile} per metric)")

    metrics = sorted({r["Metric"] for r in rows})
    df = alt.Data(values=rows)

    # Build the grid manually: hconcat over metrics, vconcat over runs within each
    # metric. Empty (Run, Metric) cells are rendered as blank to keep rows aligned.
    # x is shared within a metric column but independent across metrics.
    cell_w, cell_h = 240, 60
    has_data = {(r["Run"], r["Metric"]) for r in rows}

    columns = []
    for ci, metric in enumerate(metrics):
        is_first_col = (ci == 0)
        cells = []
        for label in labels:
            if (label, metric) in has_data:
                density_kwargs = {
                    "as_": ["Value", "density"],
                    "groupby": ["Run"],
                }
                if absolute:
                    density_kwargs["counts"] = True
                if metric in x_clip:
                    x_lo = ref_lower[metric] if absolute else 0
                    x_scale = alt.Scale(domain=[x_lo, x_clip[metric]])
                else:
                    x_scale = alt.Undefined
                cell = alt.Chart(df).transform_filter(
                    (alt.datum.Metric == metric) & (alt.datum.Run == label)
                ).transform_density(
                    "Value", **density_kwargs,
                ).mark_area(opacity=0.85, clip=True).encode(
                    x=alt.X("Value:Q",
                            title=metric if label == labels[-1] else None,
                            scale=x_scale,
                            axis=alt.Axis(labels=(label == labels[-1]),
                                          ticks=(label == labels[-1]))),
                    y=alt.Y("density:Q", stack="center", title=None,
                            axis=alt.Axis(labels=False, ticks=False, grid=False)),
                    color=alt.Color("Run:N", title="Run", sort=labels,
                                    legend=alt.Legend() if (ci == len(metrics) - 1
                                                            and label == labels[0])
                                                          else None),
                ).properties(width=cell_w, height=cell_h)
            else:
                # Blank placeholder so the row stays aligned across columns.
                cell = alt.Chart(alt.Data(values=[{"x": 0}])).mark_text(
                    text="(no data)", color="#999", fontSize=10,
                ).encode(x=alt.value(cell_w / 2), y=alt.value(cell_h / 2)
                ).properties(width=cell_w, height=cell_h)
            cells.append(cell)
        # In absolute mode share y per column too, so volumes are comparable across runs.
        col_chart = alt.vconcat(*cells, spacing=4).resolve_scale(
            x="shared", y=("shared" if absolute else "independent"),
        )
        columns.append(col_chart)

    # Row labels live in a leftmost label column.
    label_cells = [
        alt.Chart(alt.Data(values=[{"t": label}])).mark_text(
            align="right", baseline="middle", fontSize=11,
        ).encode(text="t:N", x=alt.value(120), y=alt.value(cell_h / 2)
        ).properties(width=130, height=cell_h)
        for label in labels
    ]
    label_col = alt.vconcat(*label_cells, spacing=4)

    chart = alt.hconcat(label_col, *columns, spacing=8).resolve_scale(
        color="shared",
    ).properties(
        title=(f"Program Complexity (counts; programs above DafnyBench p{lower_percentile} per metric)"
               if absolute else
               "Program Complexity (per-program mean, verified programs)"),
    )

    output_name = f"{name}-absolute" if absolute else name
    save_chart(chart, output_name)


# ---------------------------------------------------------------------------
# Plot: Program Diversity
# ---------------------------------------------------------------------------

def plot_program_diversity(
    agenda_paths: list[str],
    language: Language,
    labels: list[str] | None = None,
    name: str = "program-diversity",
) -> None:
    import altair as alt

    if labels is None:
        config = load_plots_config()
        labels = [get_run_name(config, p) for p in agenda_paths]
    backend = language.get_backend()

    # Per-feature-type rows for each diversity metric.
    entropy_rows = []
    unique_rows = []
    pnew_rows = []        # P(any single observed feature is novel) = unique / total
    pnew_prog_rows = []   # P(program contains >=1 feature never seen in earlier programs)

    for path, label in zip(agenda_paths, labels):
        agenda = load_agenda(path)
        programs = _extract_verified_programs(agenda, language)
        print(f"  {label}: {len(programs)} verified programs")

        # Per-program feature_sets, in extraction order (used for streaming P(>=1 new)).
        per_prog_features: list[dict[str, Counter]] = []
        for prog in programs:
            try:
                ft = backend.feature_sets(prog)
            except Exception:
                continue
            per_prog_features.append(ft)

        # Pooled counters for entropy / unique / P(new).
        pooled: dict[str, Counter] = {}
        for ft in per_prog_features:
            for metric, counter in ft.items():
                pooled.setdefault(metric, Counter()).update(counter)

        for metric, counter in pooled.items():
            total = sum(counter.values())
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
            pnew_rows.append({
                "Run": label,
                "Feature": metric,
                "P(new)": (len(counter) / total) if total > 0 else 0.0,
            })

        # Streaming P(>=1 new): iterate programs in order, maintain a "seen" set per
        # feature type, and count programs whose feature set introduces at least one
        # never-seen feature.
        seen: dict[str, set] = {}
        n_with_new: dict[str, int] = {}
        n_progs_with_metric: dict[str, int] = {}
        for ft in per_prog_features:
            for metric, counter in ft.items():
                if not counter:
                    continue
                n_progs_with_metric[metric] = n_progs_with_metric.get(metric, 0) + 1
                metric_seen = seen.setdefault(metric, set())
                introduced_new = False
                for feat in counter:
                    if feat not in metric_seen:
                        introduced_new = True
                        metric_seen.add(feat)
                if introduced_new:
                    n_with_new[metric] = n_with_new.get(metric, 0) + 1

        for metric, denom in n_progs_with_metric.items():
            pnew_prog_rows.append({
                "Run": label,
                "Feature": metric,
                "P(>=1 new)": (n_with_new.get(metric, 0) / denom) if denom > 0 else 0.0,
            })

    if not entropy_rows:
        print("No diversity data found.")
        return

    def _bar_chart(rows, y_field, title, fname):
        df = alt.Data(values=rows)
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Run:N", axis=None),
            y=alt.Y(f"{y_field}:Q"),
            color=alt.Color("Run:N", title="Run"),
        ).properties(width=120).facet(
            column=alt.Column("Feature:N", title="Feature"),
            spacing=10,
        ).resolve_scale(y="independent").properties(title=title)
        save_chart(chart, fname)

    _bar_chart(
        entropy_rows, "Entropy (bits)",
        "Program Diversity: Entropy (verified programs, longest per idea)",
        f"{name}-entropy",
    )
    _bar_chart(
        unique_rows, "Unique Features",
        "Program Diversity: Unique Features (verified programs, longest per idea)",
        f"{name}-unique",
    )
    _bar_chart(
        pnew_rows, "P(new)",
        "Program Diversity: P(new) = unique / total observations (verified programs)",
        f"{name}-pnew",
    )
    _bar_chart(
        pnew_prog_rows, "P(>=1 new)",
        "Program Diversity: P(>=1 new) per program (streaming, verified programs)",
        f"{name}-pnew-program",
    )


# ---------------------------------------------------------------------------
# Plot: Fixer pass@k
# ---------------------------------------------------------------------------

def _parse_label_path(spec: str) -> tuple[str, str]:
    """Parse 'Label:path' into (label, path)."""
    if ":" not in spec:
        raise ValueError(f"Expected 'Label:path', got: {spec}")
    label, path = spec.split(":", 1)
    return label.strip(), path.strip()


def plot_fixer_pass_at_k(
    horizontal: list[tuple[str, str]],
    curve: list[tuple[str, str]],
    ks: list[int],
    name: str = "fixer-pass-at-k",
) -> None:
    """Pass@k line plot. `horizontal` entries get a flat line at pass@1.
    `curve` entries get a line+points at the specified ks."""
    import altair as alt

    rows = []
    x_min, x_max = min(ks), max(ks)

    for label, path in horizontal:
        results = json.load(open(path))["results"]
        n = len(results)
        n_pass1 = sum(1 for r in results if r["success"] and r["num_attempts"] <= 1)
        rate = n_pass1 / n if n else 0.0
        rows.append({"Model": label, "k": x_min, "Pass@k": rate, "Kind": "horizontal"})
        rows.append({"Model": label, "k": x_max, "Pass@k": rate, "Kind": "horizontal"})

    for label, path in curve:
        results = json.load(open(path))["results"]
        n = len(results)
        for k in ks:
            n_pass = sum(1 for r in results if r["success"] and r["num_attempts"] <= k)
            rate = n_pass / n if n else 0.0
            rows.append({"Model": label, "k": k, "Pass@k": rate, "Kind": "curve"})

    df = alt.Data(values=rows)

    base = alt.Chart(df).encode(
        x=alt.X("k:Q",
                scale=alt.Scale(type="log", base=2, domain=[x_min, x_max]),
                axis=alt.Axis(values=ks, title="Attempts (k)")),
        y=alt.Y("Pass@k:Q", scale=alt.Scale(domain=[0, 1]), title="Pass@k"),
        color=alt.Color("Model:N", title="Model"),
    )

    line = base.mark_line().encode(
        strokeDash=alt.StrokeDash("Kind:N",
                                   scale=alt.Scale(domain=["horizontal", "curve"],
                                                   range=[[4, 4], [1, 0]]),
                                   legend=None),
    )
    points = base.transform_filter(alt.datum.Kind == "curve").mark_point(
        filled=True, size=80,
    )

    chart = (line + points).properties(
        width=400, height=300,
        title="Fixer Pass@k",
    )
    save_chart(chart, name)


# ---------------------------------------------------------------------------
# Entropy report (per-agenda program feature distributions)
# ---------------------------------------------------------------------------

def entropy_report(
    agenda_paths: list[str],
    language: Language | None = None,
    labels: list[str] | None = None,
    source: str = "verified",
) -> None:
    """Print per-feature entropy (and supporting counts) for each agenda.

    Sources:
      - "verified": longest verified program per source idea (matches the
        plotting helpers and the way the iterative SFT corpus is built).
      - "dataset":  every dataset/ object as-is (matches diversity_complexity_table).

    Useful as a quick numerical readout to track how entropy maximization
    via iterative SFT is moving the corpus over time.
    """
    if labels is None:
        config = load_plots_config()
        labels = [get_run_name(config, p) for p in agenda_paths]

    per_agenda: dict[str, dict] = {}
    for path, label in zip(agenda_paths, labels):
        agenda = load_agenda(path)
        lang = language or _detect_language(agenda)
        backend = lang.get_backend()

        if source == "verified":
            programs = _extract_verified_programs(agenda, lang)
            features: list[dict[str, Counter]] = []
            for prog in programs:
                try:
                    features.append(backend.feature_sets(prog))
                except Exception:
                    continue
            n_total = len(programs)
        elif source == "dataset":
            dataset_objs = [
                o for k, o in agenda["objects"].items() if k.startswith("dataset/")
            ]
            features = []
            for o in dataset_objs:
                text = o.content.decode("utf-8") if isinstance(o.content, bytes) else o.content
                try:
                    features.append(backend.feature_sets(Program(text, lang)))
                except Exception:
                    continue
            n_total = len(dataset_objs)
        else:
            raise ValueError(f"unknown source: {source}")

        pooled: dict[str, Counter] = {}
        for ft in features:
            for metric, counter in ft.items():
                pooled.setdefault(metric, Counter()).update(counter)

        per_agenda[label] = {
            "n_total": n_total,
            "n_parseable": len(features),
            "pooled": pooled,
        }

    print("=" * 80)
    print(f"ENTROPY (source: {source})")
    print("=" * 80)

    col_w = max(20, *(len(l) for l in labels)) + 2
    header = f"{'Metric':<40}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))

    rows: list[tuple[str, list[str]]] = []
    rows.append(("Programs", [str(per_agenda[l]["n_total"]) for l in labels]))
    rows.append(("Programs (parseable)",
                 [str(per_agenda[l]["n_parseable"]) for l in labels]))

    feature_keys = sorted({m for l in labels for m in per_agenda[l]["pooled"]})

    rows.append(("", [""] * len(labels)))
    rows.append(("Entropy (bits)", [""] * len(labels)))
    for fk in feature_keys:
        vals = []
        for l in labels:
            counter = per_agenda[l]["pooled"].get(fk, Counter())
            vals.append(f"{_entropy(counter):.2f}" if counter else "-")
        rows.append((f"  {fk}", vals))

    rows.append(("", [""] * len(labels)))
    rows.append(("Unique features", [""] * len(labels)))
    for fk in feature_keys:
        vals = []
        for l in labels:
            counter = per_agenda[l]["pooled"].get(fk, Counter())
            vals.append(str(len(counter)) if counter else "-")
        rows.append((f"  {fk}", vals))

    rows.append(("", [""] * len(labels)))
    rows.append(("Total observations", [""] * len(labels)))
    for fk in feature_keys:
        vals = []
        for l in labels:
            counter = per_agenda[l]["pooled"].get(fk, Counter())
            vals.append(str(sum(counter.values())) if counter else "-")
        rows.append((f"  {fk}", vals))

    for metric, vals in rows:
        print(f"{metric:<40}" + "".join(f"{v:>{col_w}}" for v in vals))
    print()


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
    p_tsr.add_argument("--name", default="task-success-rate",
                       help="Output filename base (default: task-success-rate)")
    p_tsr.add_argument("--labels", default=None,
                       help="Comma-separated display labels matching agenda order")

    # plot-program-complexity
    p_cx = subparsers.add_parser("plot-program-complexity",
                                 help="Box plot of complexity metrics per run")
    p_cx.add_argument("agendas", nargs="+", help="Paths to agenda .pkl files")
    p_cx.add_argument("--language", default="dafny", help="Language (default: dafny)")
    p_cx.add_argument("--name", default="program-complexity",
                      help="Output filename base (default: program-complexity)")
    p_cx.add_argument("--labels", default=None,
                      help="Comma-separated display labels matching agenda order")
    p_cx.add_argument("--dafnybench", default=None,
                      help="Path to DafnyBench ground_truth dir; adds reference rule per metric")
    p_cx.add_argument("--absolute", action="store_true",
                      help="Count-scaled violins, filtered to values above DafnyBench's "
                           "lower-percentile threshold (see --lower-percentile). "
                           "Output goes to <name>-absolute.{svg,png}. Requires --dafnybench.")
    p_cx.add_argument("--lower-percentile", type=float, default=25.0,
                      help="DafnyBench percentile used as the lower threshold in --absolute "
                           "mode and as the x-axis lower bound when also clipping (default: 25).")
    p_cx.add_argument("--x-clip-percentile", type=float, default=None,
                      help="Clip x-axis upper bound at this percentile of DafnyBench's "
                           "per-program means (e.g. 90). Requires --dafnybench.")

    # plot-fixer-pass-at-k
    p_pak = subparsers.add_parser("plot-fixer-pass-at-k",
                                  help="Line plot of fixer pass@k across models")
    p_pak.add_argument("--horizontal", nargs="*", default=[],
                       help="'Label:path' entries plotted as horizontal lines at pass@1")
    p_pak.add_argument("--curve", nargs="*", default=[],
                       help="'Label:path' entries plotted as line+points across --ks")
    p_pak.add_argument("--ks", default="1,2,4,8,16",
                       help="Comma-separated k values for curve entries (default: 1,2,4,8,16)")
    p_pak.add_argument("--name", default="fixer-pass-at-k",
                       help="Output filename base (default: fixer-pass-at-k)")

    # entropy (numerical entropy/diversity report over an agenda)
    p_ent = subparsers.add_parser(
        "entropy",
        help="Print per-feature entropy and unique-feature counts for one or "
             "more agenda checkpoints",
    )
    p_ent.add_argument("agendas", nargs="+", help="Paths to agenda .pkl files")
    p_ent.add_argument("--language", default=None,
                       help="Language (default: auto-detect from agenda objects)")
    p_ent.add_argument("--labels", default=None,
                       help="Comma-separated display labels matching agenda order")
    p_ent.add_argument("--source", default="verified",
                       choices=["verified", "dataset"],
                       help="Programs to include: 'verified' uses longest "
                            "verified program per source idea (default); "
                            "'dataset' uses every dataset/ object as-is.")

    # plot-program-diversity
    p_div = subparsers.add_parser("plot-program-diversity",
                                  help="Bar charts of diversity metrics per run")
    p_div.add_argument("agendas", nargs="+", help="Paths to agenda .pkl files")
    p_div.add_argument("--language", default="dafny", help="Language (default: dafny)")
    p_div.add_argument("--name", default="program-diversity",
                       help="Output filename base (default: program-diversity)")
    p_div.add_argument("--labels", default=None,
                       help="Comma-separated display labels matching agenda order")

    args = parser.parse_args()

    # Default to "tables" when no subcommand given (legacy behavior)
    if args.command is None:
        args = p_tables.parse_args(sys.argv[1:])
        args.command = "tables"

    if args.command == "plot-task-success-rates":
        labels = resolve_labels(args.agendas, args.labels)
        plot_task_success_rates(args.agendas, labels=labels, name=args.name)
        return

    if args.command == "plot-fixer-pass-at-k":
        horizontal = [_parse_label_path(s) for s in args.horizontal]
        curve = [_parse_label_path(s) for s in args.curve]
        ks = [int(x) for x in args.ks.split(",")]
        plot_fixer_pass_at_k(horizontal, curve, ks, name=args.name)
        return

    if args.command == "entropy":
        labels = resolve_labels(args.agendas, args.labels)
        lang = Language[args.language.upper()] if args.language else None
        entropy_report(args.agendas, language=lang, labels=labels, source=args.source)
        return

    if args.command in ("plot-program-complexity", "plot-program-diversity"):
        lang = Language[args.language.upper()]
        labels = resolve_labels(args.agendas, args.labels)
        if args.command == "plot-program-complexity":
            plot_program_complexity(args.agendas, lang, labels=labels, name=args.name,
                                    dafnybench_dir=args.dafnybench,
                                    absolute=args.absolute,
                                    x_clip_percentile=args.x_clip_percentile,
                                    lower_percentile=args.lower_percentile)
        else:
            plot_program_diversity(args.agendas, lang, labels=labels, name=args.name)
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
