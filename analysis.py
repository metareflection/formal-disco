#!/usr/bin/env python3
"""Compare multiple agenda runs: success rates and feature-set metrics."""

import argparse
import json
import math
import pickle
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

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


def _safe_percentile(xs, q):
    return float(np.percentile(xs, q)) if xs else 0.0


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


_WORKER_LANG: Language | None = None
_WORKER_BACKEND = None


def _init_feature_worker(lang: Language) -> None:
    global _WORKER_LANG, _WORKER_BACKEND
    _WORKER_LANG = lang
    _WORKER_BACKEND = lang.get_backend()


def _compute_features_in_worker(content):
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    try:
        return _WORKER_BACKEND.feature_sets(Program(text, _WORKER_LANG))
    except Exception:
        return None


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

        # Deduplicate by parent_idea: keep the longest program per lineage.
        # Objects without a parent_idea bucket on their own key (so each stands alone).
        by_idea: dict[str, tuple[str, object]] = {}
        for k, o in dataset_objs.items():
            idea = o.properties.get("parent_idea") or k
            cur = by_idea.get(idea)
            if cur is None or len(o.content) > len(cur[1].content):
                by_idea[idea] = (k, o)
        dataset_objs = {k: o for k, o in by_idea.values()}

        lang = _detect_language(a)

        contents = [o.content for o in dataset_objs.values()]
        with ProcessPoolExecutor(
            max_workers=8,
            initializer=_init_feature_worker,
            initargs=(lang,),
        ) as pool:
            results = list(tqdm(
                pool.map(_compute_features_in_worker, contents, chunksize=8),
                total=len(contents),
            ))
        all_features = [ft for ft in results if ft is not None]
        n_parseable = len(all_features)

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
        rows.append(("NUMERIC FEATURES (across per-program means)", [""] * len(labels)))
        for ck in numeric_keys:
            per_agenda_means: dict[str, list[float] | None] = {}
            for label in labels:
                d = per_agenda[label]
                if not d or not d["features"]:
                    per_agenda_means[label] = None
                    continue
                per_agenda_means[label] = [
                    _safe_mean(list(ft[ck].elements()))
                    for ft in d["features"] if ck in ft and ft[ck]
                ]

            rows.append((f"  {ck}", [""] * len(labels)))
            for stat_label, stat_fn in (
                ("mean", _safe_mean),
                ("p90", lambda xs: _safe_percentile(xs, 90)),
                ("p95", lambda xs: _safe_percentile(xs, 95)),
            ):
                row_vals = []
                for label in labels:
                    xs = per_agenda_means[label]
                    if xs is None:
                        row_vals.append("-")
                    else:
                        row_vals.append(f"{stat_fn(xs):.2f}")
                rows.append((f"    {stat_label}", row_vals))

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


def save_chart(chart, name: str, out_dir: Path = PLOTS_DIR) -> None:
    """Save an Altair chart as both .svg and .png in `out_dir` (default: plots/)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{name}.svg"
    png_path = out_dir / f"{name}.png"
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

def _extract_verified_texts(agenda: dict) -> list[str]:
    """Extract verified program *texts* from an agenda, keeping the longest per idea.

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
    return [max(texts, key=len) for texts in by_root.values()]


def _extract_verified_programs(agenda: dict, language: Language) -> list[Program]:
    """Like _extract_verified_texts, but wraps each text in a Program."""
    return [Program(text, language) for text in _extract_verified_texts(agenda)]


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


def _parse_lang_label_path(spec: str) -> tuple[str, str, str]:
    """Parse 'Language:Label:path' into (language, label, path)."""
    if spec.count(":") < 2:
        raise ValueError(f"Expected 'Language:Label:path', got: {spec}")
    lang, rest = spec.split(":", 1)
    label, path = _parse_label_path(rest)
    return lang.strip(), label, path


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (default z=1.96).

    Pass@k here is the mean over `n` independent problems of a Bernoulli
    indicator (solved within k attempts), so this is the natural CI for it.
    Behaves well at the extremes, unlike the normal (Wald) approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


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


def plot_fixer_pass_at_k_faceted(
    horizontal: list[tuple[str, str, str]],
    curve: list[tuple[str, str, str]],
    ks: list[int],
    name: str = "fixer-pass-at-k-faceted",
    title: str = "Success rate at verification annotation task",
) -> None:
    """Pass@k line plot faceted by language.

    `horizontal` entries (e.g. Claude one-shot baselines) are drawn as a flat
    dashed line at pass@1. `curve` entries get a line+points across `ks` with a
    95% Wilson confidence band. Each entry is (language, label, path); shared
    labels across languages keep colors consistent between facets."""
    import altair as alt

    rows = []
    x_min, x_max = min(ks), max(ks)

    for lang, label, path in horizontal:
        results = json.load(open(path))["results"]
        n = len(results)
        n_pass1 = sum(1 for r in results if r["success"] and r["num_attempts"] <= 1)
        rate = n_pass1 / n if n else 0.0
        for k in (x_min, x_max):
            rows.append({"Language": lang, "Model": label, "k": k,
                         "Pass@k": rate, "ci_low": rate, "ci_high": rate,
                         "Kind": "horizontal"})

    for lang, label, path in curve:
        results = json.load(open(path))["results"]
        n = len(results)
        # Only plot k values the eval actually ran: pass@k is flat once k
        # exceeds the largest num_attempts, so extending past it would draw a
        # misleading horizontal tail (e.g. a k=4 Claude curve on a k=32 axis).
        max_att = max((r["num_attempts"] for r in results), default=0)
        ks_for_file = [k for k in ks if k <= max_att] or [min(ks)]
        for k in ks_for_file:
            n_pass = sum(1 for r in results if r["success"] and r["num_attempts"] <= k)
            rate = n_pass / n if n else 0.0
            lo, hi = _wilson_interval(n_pass, n)
            rows.append({"Language": lang, "Model": label, "k": k,
                         "Pass@k": rate, "ci_low": lo, "ci_high": hi,
                         "Kind": "curve"})

    df = alt.Data(values=rows)
    langs = list(dict.fromkeys(lang for lang, _, _ in horizontal + curve))

    x = alt.X("k:Q",
              scale=alt.Scale(type="log", base=2, domain=[x_min, x_max]),
              axis=alt.Axis(values=ks, title="Attempts (k)"))
    y = alt.Y("Pass@k:Q", scale=alt.Scale(domain=[0, 1]),
              title="Success rate")
    # Legend lives inside the first (top-left) facet to save horizontal space.
    color = alt.Color("Model:N", title=None,
                      legend=alt.Legend(orient="top-left", fillColor="white",
                                        padding=8, cornerRadius=4,
                                        labelFontSize=18, symbolSize=220))

    band = alt.Chart().transform_filter(
        alt.datum.Kind == "curve"
    ).mark_area(opacity=0.15).encode(
        x=x,
        y=alt.Y("ci_low:Q", scale=alt.Scale(domain=[0, 1]), title="Success rate"),
        y2="ci_high:Q",
        color=color,
    )
    line = alt.Chart().mark_line(strokeWidth=3).encode(
        x=x, y=y, color=color,
        strokeDash=alt.StrokeDash(
            "Kind:N",
            scale=alt.Scale(domain=["horizontal", "curve"], range=[[4, 4], [1, 0]]),
            legend=None),
    )
    points = alt.Chart().transform_filter(
        alt.datum.Kind == "curve"
    ).mark_point(filled=True, size=110).encode(x=x, y=y, color=color)

    chart = alt.layer(band, line, points, data=df).properties(
        width=360, height=340,
    ).facet(
        column=alt.Column("Language:N", title=None, sort=langs),
    ).properties(
        title=title
    ).resolve_scale(x="shared", y="shared").configure_axis(
        labelFontSize=18, titleFontSize=20,
    ).configure_header(
        labelFontSize=22, titleFontSize=22,
    ).configure_title(
        fontSize=24, anchor="middle",
    ).configure_legend(
        labelFontSize=18, titleFontSize=18, labelLimit=400, symbolOpacity=1,
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
# Plot: SFT / entropy-maximization iterations
#
# Tracks how the distribution of the generated corpus moves as we iterate
# (seed Claude runs -> SFT a model on them -> regenerate -> SFT again ...).
# Iteration 0 is the seed: BOTH Claude runs (opus + sonnet) pooled together.
# ---------------------------------------------------------------------------

DEFAULT_AGENDAS_DIR = Path(__file__).parent / "final-runs" / "agendas"

# Color ramp for iterations (light = early, dark = late). Extended if needed.
_ITER_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]

# Percentile markers overlaid on each ridge, with fixed colors.
_PCT_STATS = ["mean", "p90", "p95"]
_PCT_COLORS = ["#000000", "#d95f02", "#7570b3"]


def discover_sft_iterations(
    language: Language, agendas_dir: Path,
) -> list[tuple[str, list[str]]]:
    """Discover the ordered iteration -> agenda-file(s) mapping for a language.

    Convention in final-runs/agendas:
      iteration 0 (seed) = agenda-<lang>-sn-disco3__claude-*__docs__10k.pkl  (opus + sonnet)
      iteration k+1      = agenda-<lang>-sn-disco3__qwen-docs-distilled_it<k>.pkl
    """
    lang = language.name.lower()
    prefix = f"agenda-{lang}-sn-disco3__"

    claude = sorted(agendas_dir.glob(f"{prefix}claude-*__docs__10k.pkl"))
    if not claude:
        raise FileNotFoundError(
            f"No seed (claude) agendas found: {agendas_dir}/{prefix}claude-*__docs__10k.pkl"
        )

    qwen = sorted(
        agendas_dir.glob(f"{prefix}qwen-docs-distilled_it*.pkl"),
        key=lambda p: int(p.stem.rsplit("_it", 1)[1]),
    )
    if not qwen:
        raise FileNotFoundError(
            f"No SFT-iteration agendas found: {agendas_dir}/{prefix}qwen-docs-distilled_it*.pkl"
        )

    iterations: list[tuple[str, list[str]]] = [
        ("Claude Seed", [str(p) for p in claude])
    ]
    for p in qwen:
        it_num = int(p.stem.rsplit("_it", 1)[1])
        iterations.append((f"QwenCoder2.5 it{it_num}", [str(p)]))
    return iterations


def _collect_iteration_features(
    iterations: list[tuple[str, list[str]]], language: Language,
) -> list[dict]:
    """For each iteration, extract verified program texts (pooled across its files)
    and compute per-program feature_sets in parallel."""
    out: list[dict] = []
    for idx, (label, paths) in enumerate(iterations):
        texts: list[str] = []
        for path in paths:
            print(f"  [iter {label}] loading {Path(path).name} ...")
            texts.extend(_extract_verified_texts(load_agenda(path)))
        print(f"  [iter {label}] {len(texts)} verified programs; extracting features ...")
        with ProcessPoolExecutor(
            max_workers=8,
            initializer=_init_feature_worker,
            initargs=(language,),
        ) as pool:
            results = list(tqdm(
                pool.map(_compute_features_in_worker, texts, chunksize=8),
                total=len(texts), desc=f"iter {label}",
            ))
        features = [ft for ft in results if ft is not None]
        out.append({
            "index": idx,
            "label": label,
            "features": features,
            "n_programs": len(texts),
            "n_parseable": len(features),
        })
    return out


def _kde_on_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian-KDE density on `grid`, normalized to integrate to ~1.

    Subsamples to bound the grid x points matrix; bandwidth via Silverman's rule.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.zeros_like(grid)
    std = float(np.std(v))
    n = v.size
    bw = 1.06 * std * n ** (-1 / 5) if std > 0 else 1.0
    if bw <= 0:
        bw = 1.0
    # Deterministic subsample to keep the kernel matrix small.
    if v.size > 20000:
        v = v[np.linspace(0, v.size - 1, 20000).astype(int)]
    diff = (grid[:, None] - v[None, :]) / bw
    dens = np.exp(-0.5 * diff ** 2).sum(axis=1) / (v.size * bw * np.sqrt(2 * np.pi))
    return dens


def plot_sft_iterations(
    language: Language,
    agendas_dir: Path = DEFAULT_AGENDAS_DIR,
    name: str = "sft-iterations",
    x_clip_percentile_min: float = 50.0,
    x_clip_percentile_max: float = 99.0,
    grid_points: int = 200,
) -> None:
    """Two kinds of plots for a language:

    1. Entropy across iterations: one colored line per feature.
    2. Per numeric feature: a ridgeline (one density per iteration) with
       mean / p90 / p95 markers overlaid so tail movement stays legible.
    """
    import altair as alt

    iters = _collect_iteration_features(
        discover_sft_iterations(language, agendas_dir), language
    )
    lang = language.name.lower()

    # --- Plot 1: entropy across iterations (one line per feature) ---------
    feature_keys = sorted({m for it in iters for ft in it["features"] for m in ft})

    feature_keys = [f for f in feature_keys if f not in ('subject_word', 'annotation_template', 'language_features')]

    entropy_rows = []
    for it in iters:
        pooled: dict[str, Counter] = {}
        for ft in it["features"]:
            for m, c in ft.items():
                pooled.setdefault(m, Counter()).update(c)
        for m in feature_keys:
            counter = pooled.get(m, Counter())
            entropy_rows.append({
                "Iteration": it["index"],
                "Feature": m,
                "Entropy (bits)": _entropy(counter),
            })

    iter_axis = alt.Axis(values=[it["index"] for it in iters], format="d")
    entropy_chart = alt.Chart(alt.Data(values=entropy_rows)).mark_line(
        point=True,
    ).encode(
        x=alt.X("Iteration:Q", title="SFT iteration (0 = Claude seed)", axis=iter_axis),
        y=alt.Y("Entropy (bits):Q"),
        color=alt.Color("Feature:N", title="Feature"),
    ).properties(
        width=480, height=320,
        title=f"{language.name.title()}: feature entropy across SFT iterations",
    )
    save_chart(entropy_chart, f"{name}-{lang}-entropy")

    # --- Plot 2: per numeric feature, ridgeline of the distribution -------
    numeric_keys = [
        fk for fk in feature_keys
        if any(_is_numeric_counter(ft.get(fk, Counter()))
               for it in iters for ft in it["features"] if fk in ft)
    ]

    cell_w, cell_h = 480, 60
    iter_color = {it["index"]: _ITER_COLORS[it["index"] % len(_ITER_COLORS)]
                  for it in iters}

    for fk in numeric_keys:
        # Pooled raw observations per iteration (each value repeated by its count).
        per_iter_vals: dict[int, np.ndarray] = {}
        for it in iters:
            vals: list[int] = []
            for ft in it["features"]:
                vals.extend(ft.get(fk, Counter()).elements())
            per_iter_vals[it["index"]] = np.asarray(vals, dtype=float)

        all_vals = np.concatenate([v for v in per_iter_vals.values() if v.size]) \
            if any(v.size for v in per_iter_vals.values()) else np.array([])
        if all_vals.size == 0:
            print(f"  {fk}: no observations in any iteration; skipping")
            continue

        x_min = float(np.percentile(all_vals, x_clip_percentile_min))
        x_max = float(np.percentile(all_vals, x_clip_percentile_max))
        if x_max <= 0:
            x_max = float(all_vals.max()) or 1.0
        grid = np.linspace(0.0, x_max, grid_points)

        # Densities + percentile markers per iteration; track global density max
        # so ridge amplitudes are comparable (shared y scale).
        density_rows: dict[int, list[dict]] = {}
        pct_rows: dict[int, list[dict]] = {}
        global_dmax = 0.0
        for it in iters:
            idx = it["index"]
            v = per_iter_vals[idx]
            if v.size == 0:
                density_rows[idx] = []
                pct_rows[idx] = []
                continue
            dens = _kde_on_grid(v, grid)
            global_dmax = max(global_dmax, float(dens.max()))
            density_rows[idx] = [
                {"value": float(g), "density": float(d)} for g, d in zip(grid, dens)
            ]
            pct_rows[idx] = [
                {"stat": "mean", "value": float(np.mean(v))},
                {"stat": "p90", "value": float(np.percentile(v, 90))},
                {"stat": "p95", "value": float(np.percentile(v, 95))},
            ]
        if global_dmax <= 0:
            global_dmax = 1.0

        x_scale = alt.Scale(domain=[x_min, x_max])
        y_scale = alt.Scale(domain=[0.0, global_dmax])

        # Stack ridges top = latest iteration, bottom = iteration 0.
        cells = []
        ordered = sorted(iters, key=lambda it: it["index"], reverse=True)
        for pos, it in enumerate(ordered):
            idx = it["index"]
            is_bottom = (pos == len(ordered) - 1)
            is_top = (pos == 0)
            x_enc = alt.X(
                "value:Q", scale=x_scale,
                title=fk if is_bottom else None,
                axis=alt.Axis(labels=is_bottom, ticks=is_bottom, grid=False),
            )

            label_text = f"iter {it['label']}  (n={it['n_programs']})"
            label_layer = alt.Chart(alt.Data(values=[{"t": label_text}])).mark_text(
                align="left", baseline="top", fontSize=10, color="#333",
            ).encode(x=alt.value(4), y=alt.value(2), text="t:N")

            if not density_rows[idx]:
                cells.append(alt.layer(
                    alt.Chart(alt.Data(values=[{"x": 0}])).mark_text(
                        text="(no data)", color="#999", fontSize=10,
                    ).encode(x=alt.value(cell_w / 2), y=alt.value(cell_h / 2)),
                    label_layer,
                ).properties(width=cell_w, height=cell_h))
                continue

            area = alt.Chart(alt.Data(values=density_rows[idx])).mark_area(
                opacity=0.8, interpolate="monotone",
                line={"color": "#333", "strokeWidth": 0.5},
            ).encode(
                x=x_enc,
                y=alt.Y("density:Q", scale=y_scale, axis=None, title=None),
                color=alt.value(iter_color[idx]),
            )
            rules = alt.Chart(alt.Data(values=pct_rows[idx])).mark_rule(
                strokeWidth=1.5,
            ).encode(
                x=alt.X("value:Q", scale=x_scale),
                color=alt.Color(
                    "stat:N",
                    scale=alt.Scale(domain=_PCT_STATS, range=_PCT_COLORS),
                    legend=(alt.Legend(title="marker") if is_top else None),
                ),
            )
            cells.append(alt.layer(area, rules, label_layer)
                         .properties(width=cell_w, height=cell_h))

        ridgeline = alt.vconcat(*cells, spacing=-int(cell_h * 0.35)).resolve_scale(
            x="shared", y="shared", color="independent",
        ).properties(
            title=f"{language.name.title()}: distribution of '{fk}' across SFT iterations "
                  f"(x from p{x_clip_percentile_min:g} to p{x_clip_percentile_max:g})",
        )
        save_chart(ridgeline, f"{name}-{lang}-dist-{fk}")


# ---------------------------------------------------------------------------
# Plot: per-iteration task success rates + feature entropy (line plots)
#
# Both share the same iteration discovery/labelling as plot_sft_iterations:
# the seed (opus + sonnet pooled) is "Claude Seed"; SFT iterations are it0, it1, ...
# ---------------------------------------------------------------------------

def _collect_iteration_data(
    iterations: list[tuple[str, list[str]]], language: Language,
) -> list[dict]:
    """Load each iteration's agenda(s) ONCE, returning both the verified-program
    feature_sets and the pooled task_outcomes per iteration.

    The seed iteration pools across its multiple files (opus + sonnet); task
    outcome counts are summed per (task type, work status).
    """
    out: list[dict] = []
    for idx, (label, paths) in enumerate(iterations):
        texts: list[str] = []
        task_outcomes: dict[str, Counter] = {}
        for path in paths:
            print(f"  [iter {label}] loading {Path(path).name} ...")
            agenda = load_agenda(path)
            texts.extend(_extract_verified_texts(agenda))
            for tt, oc in agenda.get("task_outcomes", {}).items():
                task_outcomes.setdefault(tt, Counter()).update(oc)
        print(f"  [iter {label}] {len(texts)} verified programs; extracting features ...")
        with ProcessPoolExecutor(
            max_workers=8,
            initializer=_init_feature_worker,
            initargs=(language,),
        ) as pool:
            results = list(tqdm(
                pool.map(_compute_features_in_worker, texts, chunksize=8),
                total=len(texts), desc=f"iter {label}",
            ))
        features = [ft for ft in results if ft is not None]
        out.append({
            "index": idx,
            "label": label,
            "features": features,
            "n_programs": len(texts),
            "n_parseable": len(features),
            "task_outcomes": task_outcomes,
        })
    return out


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (center - half, center + half)


# Friendlier, "professional" task labels (and a stable display order).
_TASK_LABELS = {
    "initiate-program": "Initiate",
    "repair": "Repair",
    "extend": "Extend",
}
_TASK_ORDER = ["Initiate", "Repair", "Extend"]

# Font sizes, ~50% larger than the Vega-Lite defaults.
_FONT_AXIS_LABEL = 15
_FONT_AXIS_TITLE = 17
_FONT_LEGEND_LABEL = 15
_FONT_LEGEND_TITLE = 16
_FONT_HEADER = 18
_FONT_TITLE = 21
_FONT_SUBTITLE = 15


def _configure_iteration_fonts(chart):
    """Apply the (enlarged) font sizes shared by the iteration plots."""
    return (
        chart
        .configure_axis(labelFontSize=_FONT_AXIS_LABEL, titleFontSize=_FONT_AXIS_TITLE)
        .configure_legend(labelFontSize=_FONT_LEGEND_LABEL,
                          titleFontSize=_FONT_LEGEND_TITLE,
                          labelLimit=0)
        .configure_header(labelFontSize=_FONT_HEADER, titleFontSize=_FONT_HEADER,
                          labelFontWeight="bold")
        .configure_title(fontSize=_FONT_TITLE, subtitleFontSize=_FONT_SUBTITLE,
                         anchor="middle")
    )


def plot_iteration_metrics(
    languages: list[Language],
    agendas_dir: Path = DEFAULT_AGENDAS_DIR,
    out_dir: Path = DEFAULT_AGENDAS_DIR.parent,
    name_prefix: str = "iterations",
    cache: Path | None = None,
) -> None:
    """Two line plots, faceted by language, across SFT iterations (x = 1, 2, ...,
    where 1 is the first Qwen run). The Claude seed runs are shown as horizontal
    dashed reference lines instead of an iteration of their own.

      1. Task success rate, one colored line per task type, with 95% Wilson
         CIs (error bars on iterations, a faint band around the seed line).
      2. Feature entropy, one colored line per optimized feature
         (all metrics in backend.feature_metrics -- the features ranked by
         surprisal when building the iterative-SFT corpus in distill.py).
    """
    import altair as alt

    # The metrics we optimize for: the numeric structural features
    # (feature_metrics minus subject_word / annotation_template / language_features).
    _NON_OPTIMIZED = {"subject_word", "annotation_template", "language_features"}

    def _success_rows(it, iteration, lang_title):
        rows = []
        for task_type, oc in it["task_outcomes"].items():
            done = oc.get("DONE", 0)
            total = done + oc.get("FAILED", 0) + oc.get("ATTEMPTED", 0)
            if total == 0:
                continue
            lo, hi = _wilson_ci(done, total)
            rows.append({
                "Language": lang_title,
                "Iteration": iteration,
                "Task": _TASK_LABELS.get(task_type, task_type),
                "Success Rate": done / total,
                "CI low": lo,
                "CI high": hi,
                "n": total,
            })
        return rows

    def _entropy_rows(it, iteration, lang_title, feature_keys):
        pooled: dict[str, Counter] = {}
        for ft in it["features"]:
            for m, c in ft.items():
                pooled.setdefault(m, Counter()).update(c)
        return [{
            "Language": lang_title,
            "Iteration": iteration,
            "Feature": m.replace("_", " ").title(),  # e.g. "Lemma Body Size"
            "Entropy (bits)": _entropy(pooled.get(m, Counter())),
        } for m in feature_keys]

    success_rows: list[dict] = []
    seed_rows: list[dict] = []
    entropy_rows: list[dict] = []
    seed_entropy_rows: list[dict] = []
    n_iters = 0

    # Loading + feature-extracting every iteration agenda is slow; allow caching
    # the (cheap, plottable) row data so plot styling can be iterated quickly.
    if cache is not None and Path(cache).exists():
        print(f"Loading cached iteration rows from {cache}")
        with open(cache, "rb") as f:
            success_rows, seed_rows, entropy_rows, seed_entropy_rows, n_iters = \
                pickle.load(f)
    else:
        for language in languages:
            lang_title = language.name.title()
            data = _collect_iteration_data(
                discover_sft_iterations(language, agendas_dir), language
            )
            seed, qwen = data[0], data[1:]
            n_iters = max(n_iters, len(qwen))

            for i, it in enumerate(qwen):
                success_rows += _success_rows(it, i + 1, lang_title)
            seed_rows += _success_rows(seed, None, lang_title)

            feature_keys = sorted(
                m for m in language.get_backend().feature_metrics
                if m not in _NON_OPTIMIZED
            )
            for i, it in enumerate(qwen):
                entropy_rows += _entropy_rows(it, i + 1, lang_title, feature_keys)
            seed_entropy_rows += _entropy_rows(seed, None, lang_title, feature_keys)

        if cache is not None:
            with open(cache, "wb") as f:
                pickle.dump((success_rows, seed_rows, entropy_rows,
                             seed_entropy_rows, n_iters), f)
            print(f"Cached iteration rows to {cache}")

    # x-axis shared by both plots: integer iterations 1..N, with a little padding
    # so the first/last markers are not clipped against the panel edges.
    x_axis = alt.X(
        "Iteration:Q", title="SFT iteration",
        axis=alt.Axis(values=list(range(1, n_iters + 1)), format="d"),
        scale=alt.Scale(domain=[0.8, n_iters + 0.2], nice=False),
    )
    # Legend placed inside the first panel to save horizontal space. A corner
    # `orient` is dropped when faceting, so position it explicitly (coordinates
    # are relative to the whole concatenated view).
    def _inside_legend(legend_x, legend_y, direction="vertical", columns=None):
        # symbolOpacity=1 keeps swatches solid (the faint seed-band layer shares
        # this legend and would otherwise wash the symbols out).
        extra = {} if columns is None else {"columns": columns}
        return alt.Legend(orient="none", legendX=legend_x, legendY=legend_y,
                          direction=direction, labelLimit=300,
                          fillColor="white", padding=7,
                          strokeColor="lightgray", cornerRadius=4,
                          symbolOpacity=1, symbolStrokeWidth=0, symbolSize=170,
                          **extra)
    facet = alt.Facet("Language:N", title=None,
                      sort=[l.name.title() for l in languages])

    # --- Plot 1: task success rate (DONE / total attempts) per task type ---
    if not success_rows:
        print("No task_outcome data found; skipping task success-rate plot.")
    else:
        y_max = max(
            [r["CI high"] for r in success_rows]
            + [r["CI high"] for r in seed_rows]
            + [r["Success Rate"] for r in success_rows + seed_rows]
        ) * 1.05
        # All layers must share the SAME legend object: a sibling layer with
        # legend=None would otherwise suppress the (shared) faceted legend.
        color = alt.Color("Task:N", title="Task", sort=_TASK_ORDER,
                          legend=_inside_legend(240, 100))
        # Faceting a layered chart needs a single top-level dataset; tag each
        # row's source so the per-mark layers can filter to it.
        success_data = (
            [{**r, "Source": "iter"} for r in success_rows]
            + [{**r, "Source": "seed"} for r in seed_rows]
        )
        base = alt.Chart()
        is_iter = base.transform_filter(alt.datum.Source == "iter")
        is_seed = base.transform_filter(alt.datum.Source == "seed")
        line = is_iter.mark_line(point=True).encode(
            x=x_axis,
            y=alt.Y("Success Rate:Q", title="Success rate",
                    scale=alt.Scale(domain=[0, y_max])),
            color=color,
            tooltip=["Language:N", "Iteration:Q", "Task:N", "Success Rate:Q",
                     "CI low:Q", "CI high:Q", "n:Q"],
        )
        errors = is_iter.mark_errorbar(ticks=True).encode(
            x=x_axis,
            y=alt.Y("CI low:Q", title="Success rate"),
            y2="CI high:Q",
            color=color,
        )
        seed_band = is_seed.mark_rect(opacity=0.15).encode(
            y=alt.Y("CI low:Q", title="Success rate"),
            y2="CI high:Q",
            color=color,
        )
        seed_rule = is_seed.mark_rule(strokeDash=[6, 4], strokeWidth=1.5).encode(
            y=alt.Y("Success Rate:Q"),
            color=color,
        )
        success_chart = alt.layer(
            seed_band, seed_rule, errors, line,
            data=alt.Data(values=success_data),
        ).properties(width=320, height=320).facet(facet).properties(
            title="Task success rates across SFT iterations",
        )
        save_chart(_configure_iteration_fonts(success_chart),
                   f"{name_prefix}-task-success-rate", out_dir)

    # --- Plot 2: feature entropy, one line per optimized feature ----------
    y_max_e = max(
        [r["Entropy (bits)"] for r in entropy_rows + seed_entropy_rows]
    ) * 1.05
    entropy_data = (
        [{**r, "Source": "iter"} for r in entropy_rows]
        + [{**r, "Source": "seed"} for r in seed_entropy_rows]
    )
    # Single-row legend inside the empty bottom band (above the x-axis, below
    # the lowest line).
    ecolor = alt.Color(
        "Feature:N", title="Feature",
        legend=alt.Legend(orient="none", legendX=30, legendY=265,
                          direction="horizontal", labelLimit=300,
                          fillColor="white", padding=5,
                          strokeColor="lightgray", cornerRadius=4,
                          symbolOpacity=1, symbolStrokeWidth=0, symbolSize=170)
    )
    ebase = alt.Chart()
    entropy_lines = ebase.transform_filter(
        alt.datum.Source == "iter",
    ).mark_line(point=True).encode(
        x=x_axis,
        y=alt.Y("Entropy (bits):Q", scale=alt.Scale(domain=[0, y_max_e])),
        color=ecolor,
        tooltip=["Language:N", "Iteration:Q", "Feature:N", "Entropy (bits):Q"],
    )
    seed_entropy_rules = ebase.transform_filter(
        alt.datum.Source == "seed",
    ).mark_rule(strokeDash=[6, 4], strokeWidth=1.5).encode(
        y=alt.Y("Entropy (bits):Q"),
        color=ecolor,
    )
    entropy_chart = alt.layer(
        seed_entropy_rules, entropy_lines,
        data=alt.Data(values=entropy_data),
    ).properties(width=320, height=320).facet(facet).properties(
        title="Feature entropy across SFT iterations",
    )
    save_chart(_configure_iteration_fonts(entropy_chart),
               f"{name_prefix}-feature-entropy", out_dir)


# ---------------------------------------------------------------------------
# Plot: rarefaction curves (feature entropy vs. number of programs sampled)
#
# A rarefaction curve shows how the entropy of a feature's distribution grows
# as more programs are drawn from a dataset. A curve that has plateaued means
# the dataset's diversity in that feature is saturated; one still rising means
# more programs would keep surfacing novel structure. Faceted by feature
# (rows) x language (columns), one colored line per dataset.
# ---------------------------------------------------------------------------

class _IncrementalEntropy:
    """Shannon entropy (bits) of a growing multiset, updated in O(keys added).

    Uses the identity H = log2(T) - S/T, where T is the total count and
    S = sum_k c_k * log2(c_k). Adding observations only touches the keys whose
    counts change, so walking a rarefaction curve costs one cheap update per
    feature occurrence rather than re-pooling the whole counter at each step.
    """

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.total: int = 0
        self._s: float = 0.0  # sum_k c_k * log2(c_k)

    def add(self, counter: Counter) -> None:
        for key, delta in counter.items():
            if delta == 0:
                continue
            c0 = self.counts[key]
            c1 = c0 + delta
            self.counts[key] = c1
            if c0 > 0:
                self._s -= c0 * math.log2(c0)
            self._s += c1 * math.log2(c1)
            self.total += delta

    def entropy(self) -> float:
        if self.total == 0:
            return 0.0
        return max(0.0, math.log2(self.total) - self._s / self.total)


def _load_one_path(path: str) -> tuple[Language, list[str]]:
    """Load (language, program texts) for a single rarefaction source path.

    Supported (language is inferred, since the plot facets by language):
      - .pkl agenda checkpoint  -> longest verified program per idea
      - directory of .dfy/.rs   -> every file's contents (recursive)
    """
    p = Path(path)

    if p.is_dir():
        dfy = sorted(p.rglob("*.dfy"))
        rs = sorted(p.rglob("*.rs"))
        if dfy and rs:
            raise ValueError(
                f"{path}: directory mixes .dfy and .rs files; cannot infer language"
            )
        if dfy:
            lang, files = Language.DAFNY, dfy
        elif rs:
            lang, files = Language.VERUS, rs
        else:
            raise ValueError(f"{path}: no .dfy or .rs files found")
        return lang, [f.read_text(encoding="utf-8", errors="replace") for f in files]

    if p.suffix == ".pkl":
        agenda = load_agenda(path)
        return _detect_language(agenda), _extract_verified_texts(agenda)

    raise ValueError(f"{path}: unsupported source (expected a .pkl file or a directory)")


def _load_rarefaction_source(spec: str) -> tuple[str, Language, list[str]]:
    """Parse a 'Label:path' source spec into (label, language, program texts).

    Several paths may be pooled into one dataset by joining them with '+'
    (e.g. 'Claude Seed:opus.pkl+sonnet.pkl'); they must share a language.
    See `_load_one_path` for the per-path formats supported.
    """
    label, path = _parse_label_path(spec)

    lang: Language | None = None
    texts: list[str] = []
    for part in path.split("+"):
        part_lang, part_texts = _load_one_path(part.strip())
        if lang is not None and part_lang != lang:
            raise ValueError(
                f"{label}: pooled paths mix languages "
                f"({lang.name} vs {part_lang.name})"
            )
        lang = part_lang
        texts.extend(part_texts)
    return label, lang, texts


def _extract_features_parallel(
    texts: list[str], language: Language, desc: str = "",
    cache_dir: Path | None = None,
) -> list[dict[str, Counter]]:
    """Feature-extract `texts` in a process pool, dropping unparseable programs.

    If `cache_dir` is given, the extracted feature list is memoized on disk keyed
    by (language, the texts themselves), so repeated rarefaction runs over the
    same sources (e.g. features1 then features2) skip re-parsing.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        import hashlib
        h = hashlib.sha256()
        h.update(language.name.encode())
        h.update(str(len(texts)).encode())
        for t in texts:
            h.update(b"\x00")
            h.update(t.encode("utf-8", errors="replace"))
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{language.name.lower()}-{h.hexdigest()[:16]}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    with ProcessPoolExecutor(
        max_workers=8,
        initializer=_init_feature_worker,
        initargs=(language,),
    ) as pool:
        results = list(tqdm(
            pool.map(_compute_features_in_worker, texts, chunksize=8),
            total=len(texts), desc=desc,
        ))
    features = [ft for ft in results if ft is not None]
    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(features, f)
    return features


def _rarefaction_checkpoints(n: int, base: float = 2.0, min_s: int = 10) -> list[int]:
    """Log-spaced program counts at which to sample the curve.

    Geometric steps 1, base, base^2, ... up to (but excluding) `n`, with the
    full count `n` always appended as the last checkpoint. Starting at 1 keeps
    the left end of the (log-scaled) x-axis anchored at a single program;
    duplicate integer counts (from small `base`) are collapsed.
    """
    if n <= 0:
        return []
    cps: list[int] = []
    v = 1.0
    while v < n:
        if v >= min_s:
            cps.append(int(v))
        v *= base
    cps.append(n)
    return sorted(set(cps))


def plot_rarefaction_curves(
    sources: list[str],
    base: float = 2.0,
    metrics: list[str] | None = None,
    seed: int = 0,
    repeats: int = 1,
    name: str = "rarefaction",
    out_dir: Path = PLOTS_DIR,
    feature_cache: Path | None = None,
    statistic: str = "entropy",
) -> None:
    """Feature-entropy rarefaction curves, faceted by feature x language.

    For each dataset the programs are shuffled, then added one at a time to a
    growing pool; at log-spaced checkpoints (1, `base`, `base`^2, ..., and the
    full dataset size) the entropy of every feature's pooled distribution is
    recorded. The x-axis is log-scaled, so this reads off how saturated each
    dataset's diversity is across orders of magnitude of programs.

    With `statistic="unique"` the y-axis is instead the number of distinct
    feature values observed (a species-accumulation curve). This captures
    coverage differences that bulk entropy hides: a dataset can match another's
    entropy while whole feature values (e.g. rarely-used language features)
    never appear in it at all.

    With `repeats > 1` the sampling is redone with `repeats` distinct shuffles
    (seeds `seed`, `seed+1`, ...); the plot then shows the mean curve with a
    Vega-Lite standard-error band across shuffles.
    """
    import altair as alt

    if statistic not in ("entropy", "unique"):
        raise ValueError(f"unknown statistic: {statistic}")
    y_field = "Entropy (bits)" if statistic == "entropy" else "Unique values"

    rows: list[dict] = []

    for spec in sources:
        label, language, texts = _load_rarefaction_source(spec)
        backend_metrics = set(language.get_backend().feature_metrics)
        wanted = backend_metrics if metrics is None else backend_metrics & set(metrics)
        if not wanted:
            print(f"  [{label}] no requested metrics for {language.name}; skipping")
            continue

        print(f"  [{label}] {len(texts)} programs ({language.name}); extracting features ...")
        features = _extract_features_parallel(texts, language, desc=label,
                                              cache_dir=feature_cache)
        if not features:
            print(f"  [{label}] no parseable programs; skipping")
            continue

        # Checkpoints depend only on the dataset size, so they are identical
        # across shuffles -- each N gets `repeats` entropy samples to band over.
        checkpoints = set(_rarefaction_checkpoints(len(features), base))

        for r in range(repeats):
            rng = random.Random(seed + r)
            order = list(range(len(features)))
            rng.shuffle(order)

            accs = {m: _IncrementalEntropy() for m in wanted}
            for i, idx in enumerate(order, start=1):
                ft = features[idx]
                for m, acc in accs.items():
                    acc.add(ft.get(m, Counter()))
                if i in checkpoints:
                    for m, acc in accs.items():
                        rows.append({
                            "Dataset": label,
                            "Language": language.name.title(),
                            "Feature": m.replace("_", " ").title(),
                            "Repeat": r,
                            "N": i,
                            y_field: (acc.entropy() if statistic == "entropy"
                                      else len(acc.counts)),
                        })

    if not rows:
        print("No data to plot.")
        return

    languages = sorted({r["Language"] for r in rows})
    features_sorted = sorted({r["Feature"] for r in rows})

    base_chart = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("N:Q", title="Number of programs",
                scale=alt.Scale(type="log")),
        color=alt.Color("Dataset:N", title="Dataset"),
    )
    # zero=False lets each facet's y-range hug its data instead of wasting
    # vertical space down to 0 (entropies sit well above it).
    y_scale = alt.Scale(zero=False)
    # Standard-error band across the shuffles (zero-width when repeats == 1).
    band = base_chart.mark_errorband(extent="stderr").encode(
        y=alt.Y(f"{y_field}:Q", title=y_field, scale=y_scale),
    )
    # Mean curve over the shuffles.
    line = base_chart.mark_line(point=True).encode(
        y=alt.Y(f"mean({y_field}):Q", title=y_field, scale=y_scale),
        tooltip=["Dataset:N", "Language:N", "Feature:N", "N:Q",
                 alt.Tooltip(f"mean({y_field}):Q", title=f"Mean {y_field.lower()}")],
    )
    chart = alt.layer(band, line).properties(width=300, height=220).facet(
        row=alt.Row("Feature:N", title=None, sort=features_sorted),
        column=alt.Column("Language:N", title=None, sort=languages),
    ).resolve_scale(x="independent", y="independent").properties(
        title=("Feature-entropy rarefaction curves" if statistic == "entropy"
               else "Feature-coverage rarefaction curves"),
    )
    save_chart(_configure_iteration_fonts(chart), name, out_dir)


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

    # plot-fixer-pass-at-k-faceted
    p_pakf = subparsers.add_parser(
        "plot-fixer-pass-at-k-faceted",
        help="Fixer pass@k line plot faceted by language (Claude one-shot "
             "baselines as horizontal lines; curves get 95% Wilson CI bands)")
    p_pakf.add_argument("--horizontal", nargs="*", default=[],
                        help="'Language:Label:path' entries plotted as horizontal "
                             "lines at pass@1")
    p_pakf.add_argument("--curve", nargs="*", default=[],
                        help="'Language:Label:path' entries plotted as line+points "
                             "with a CI band across --ks")
    p_pakf.add_argument("--ks", default="1,2,4,8,16,32",
                        help="Comma-separated k values for curve entries "
                             "(default: 1,2,4,8,16,32)")
    p_pakf.add_argument("--name", default="fixer-pass-at-k-faceted",
                        help="Output filename base (default: fixer-pass-at-k-faceted)")
    p_pakf.add_argument("--title",
                        default="Success rate at verification annotation task",
                        help="Chart title")

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

    # plot-sft-iterations
    p_sft = subparsers.add_parser(
        "plot-sft-iterations",
        help="Entropy-across-iterations line plot + per-numeric-feature ridgelines "
             "showing the corpus distribution shifting under iterative SFT",
    )
    p_sft.add_argument("--language", default="dafny", help="Language (default: dafny)")
    p_sft.add_argument("--agendas-dir", default=str(DEFAULT_AGENDAS_DIR),
                       help=f"Directory of iteration agenda .pkl files "
                            f"(default: {DEFAULT_AGENDAS_DIR})")
    p_sft.add_argument("--name", default="sft-iterations",
                       help="Output filename base (default: sft-iterations)")
    p_sft.add_argument("--x-clip-percentile-min", type=float, default=50.0,
                       help="Clip ridgeline min x-axis at this percentile of pooled "
                            "observations (default: 50 )")
    p_sft.add_argument("--x-clip-percentile-max", type=float, default=99.0,
                       help="Clip ridgeline x-axis at this percentile of pooled "
                            "observations (default: 99)")

    # plot-iterations (task success rates + feature entropy, as line plots)
    p_it = subparsers.add_parser(
        "plot-iterations",
        help="Line plots across SFT iterations: task success rates and feature "
             "entropy (seed = 'Claude Seed', then it0, it1, ...)",
    )
    p_it.add_argument("--language", default="dafny,verus",
                      help="Comma-separated languages to facet over "
                           "(default: dafny,verus)")
    p_it.add_argument("--agendas-dir", default=str(DEFAULT_AGENDAS_DIR),
                      help=f"Directory of iteration agenda .pkl files "
                           f"(default: {DEFAULT_AGENDAS_DIR})")
    p_it.add_argument("--out-dir", default=str(DEFAULT_AGENDAS_DIR.parent),
                      help=f"Directory to write the plots into "
                           f"(default: {DEFAULT_AGENDAS_DIR.parent})")
    p_it.add_argument("--name-prefix", default="iterations",
                      help="Output filename prefix (default: iterations)")
    p_it.add_argument("--cache", default=None,
                      help="Optional pickle path to cache/reuse the extracted "
                           "row data, to iterate on plot styling without "
                           "re-parsing every agenda")

    # plot-program-diversity
    p_div = subparsers.add_parser("plot-program-diversity",
                                  help="Bar charts of diversity metrics per run")
    p_div.add_argument("agendas", nargs="+", help="Paths to agenda .pkl files")
    p_div.add_argument("--language", default="dafny", help="Language (default: dafny)")
    p_div.add_argument("--name", default="program-diversity",
                       help="Output filename base (default: program-diversity)")
    p_div.add_argument("--labels", default=None,
                       help="Comma-separated display labels matching agenda order")

    # plot-rarefaction
    p_rar = subparsers.add_parser(
        "plot-rarefaction",
        help="Feature-entropy rarefaction curves (entropy vs. #programs), "
             "faceted by feature x language, one line per dataset",
    )
    p_rar.add_argument("sources", nargs="+",
                       help="'Label:path' sources; path is a .pkl agenda "
                            "(verified programs) or a directory of .dfy/.rs files. "
                            "Pool several paths into one dataset with '+' "
                            "(e.g. 'Seed:opus.pkl+sonnet.pkl').")
    p_rar.add_argument("--base", type=float, default=2.0,
                       help="Log base for the x-axis checkpoints: programs are "
                            "sampled at 1, base, base^2, ... plus the full "
                            "dataset size (default: 2)")
    p_rar.add_argument("--metrics", default=None,
                       help="Comma-separated feature metrics to plot "
                            "(default: all of each language's feature_metrics)")
    p_rar.add_argument("--seed", type=int, default=0,
                       help="Base shuffle seed for the sampling order (default: 0)")
    p_rar.add_argument("--repeats", type=int, default=1,
                       help="Number of shuffles to average; >1 draws a "
                            "standard-error band across them (default: 1)")
    p_rar.add_argument("--name", default="rarefaction",
                       help="Output filename base (default: rarefaction)")
    p_rar.add_argument("--out-dir", default=str(PLOTS_DIR),
                       help=f"Directory to write the plot into (default: {PLOTS_DIR})")
    p_rar.add_argument("--feature-cache", default=None,
                       help="Directory to memoize extracted features per source, so "
                            "repeated runs over the same datasets skip re-parsing")
    p_rar.add_argument("--statistic", default="entropy",
                       choices=["entropy", "unique"],
                       help="Y-axis statistic: pooled feature entropy, or number "
                            "of distinct feature values observed (coverage; "
                            "default: entropy)")

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

    if args.command == "plot-fixer-pass-at-k-faceted":
        horizontal = [_parse_lang_label_path(s) for s in args.horizontal]
        curve = [_parse_lang_label_path(s) for s in args.curve]
        ks = [int(x) for x in args.ks.split(",")]
        plot_fixer_pass_at_k_faceted(horizontal, curve, ks, name=args.name,
                                     title=args.title)
        return

    if args.command == "entropy":
        labels = resolve_labels(args.agendas, args.labels)
        lang = Language[args.language.upper()] if args.language else None
        entropy_report(args.agendas, language=lang, labels=labels, source=args.source)
        return

    if args.command == "plot-sft-iterations":
        lang = Language[args.language.upper()]
        plot_sft_iterations(
            lang,
            agendas_dir=Path(args.agendas_dir),
            name=args.name,
            x_clip_percentile_min=args.x_clip_percentile_min,
            x_clip_percentile_max=args.x_clip_percentile_max,
        )
        return

    if args.command == "plot-iterations":
        langs = [Language[s.strip().upper()]
                 for s in args.language.split(",") if s.strip()]
        plot_iteration_metrics(
            langs,
            agendas_dir=Path(args.agendas_dir),
            out_dir=Path(args.out_dir),
            name_prefix=args.name_prefix,
            cache=Path(args.cache) if args.cache else None,
        )
        return

    if args.command == "plot-rarefaction":
        metrics = ([m.strip() for m in args.metrics.split(",") if m.strip()]
                   if args.metrics else None)
        plot_rarefaction_curves(
            args.sources,
            base=args.base,
            metrics=metrics,
            seed=args.seed,
            repeats=args.repeats,
            name=args.name,
            out_dir=Path(args.out_dir),
            feature_cache=Path(args.feature_cache) if args.feature_cache else None,
            statistic=args.statistic,
        )
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
