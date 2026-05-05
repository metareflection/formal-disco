"""One-page recap of a formal-disco run.

Reads the pickle and (optionally) the trace JSONL, prints a structured
summary covering: counts of concepts/theorems/refutations, per-heuristic
worth tracking, heuristic births (with provenance attribution), refutation
witnesses, soundness-gate verdicts, and — most importantly — provenance
from each proved theorem back to its originating heuristic.

The provenance section is what answers the Phase 2 acceptance question:
"did at least one reflection-born heuristic contribute a proved theorem?"

Usage:
    python -m cli.summarize agenda-phase2-modarith.pkl
    python -m cli.summarize agenda-phase2-modarith.pkl --jsonl agenda-phase2-modarith.trace.jsonl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional


def _hr(c: str = "-", n: int = 78) -> None:
    print(c * n)


def _pkl_path(p: str) -> Path:
    path = Path(p)
    if not path.exists():
        sys.exit(f"error: pickle not found: {path}")
    return path


def _jsonl_for(pkl: Path) -> Optional[Path]:
    """Default trace JSONL alongside the pickle, if it exists."""
    candidate = pkl.with_suffix(".trace.jsonl")
    if candidate.exists():
        return candidate
    base, _, _ = pkl.name.rpartition(".")
    candidate = pkl.parent / f"{base}.trace.jsonl"
    return candidate if candidate.exists() else None


def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _format_int(n: int) -> str:
    return f"{n:>5d}"


def overview(data: dict) -> None:
    objects = data["objects"]
    concepts = [o for o in objects.values() if o.type == "concept"]
    theorems = [o for o in concepts if o.properties.get("kind") == "theorem"]
    refuted = [o for o in concepts if o.properties.get("refuted")]
    too_sim = [o for o in concepts if o.properties.get("too_similar")]
    heuristics = [o for o in objects.values() if o.type == "heuristic"]
    seed_h = [h for h in heuristics if not h.properties.get("born_from_reflection")]
    born_h = [h for h in heuristics if h.properties.get("born_from_reflection")]
    traces = [o for o in objects.values() if o.type == "trace"]

    _hr("=")
    print("OVERVIEW")
    _hr("=")
    print(f"  clock:               {data.get('clock', '?')}")
    print(f"  total_attempts:      {data.get('total_attempts', '?')}")
    print(f"  concepts:            {len(concepts):>5d}  (theorems={len(theorems)}, refuted={len(refuted)}, too_similar={len(too_sim)})")
    print(f"  heuristics:          {len(heuristics):>5d}  ({len(seed_h)} seed + {len(born_h)} born)")
    print(f"  trace events stored: {len(traces):>5d}")


def heuristic_table(data: dict) -> None:
    """Per-heuristic worth + activity table."""
    objects = data["objects"]
    heuristics = sorted(
        [o for o in objects.values() if o.type == "heuristic"],
        key=lambda o: -float(o.interestingness),
    )
    print()
    _hr("=")
    print("HEURISTICS")
    _hr("=")
    print(f"  {'name':<40s} {'born?':<6s} {'wrth':>6s} {'att':>4s} {'adm':>4s} {'pro':>4s} {'nov':>5s} {'dif':>5s}")
    print("  " + "-" * 76)
    for h in heuristics:
        p = h.properties
        attempts = int(p.get("attempts", 0) or 0)
        admits = int(p.get("admits", 0) or 0)
        proves = int(p.get("proves", p.get("successes", 0)) or 0)
        novelty_sum = float(p.get("novelty_sum", 0.0) or 0.0)
        diff_sum = float(p.get("difficulty_sum", 0.0) or 0.0)
        nov_avg = novelty_sum / admits if admits > 0 else 0.0
        diff_avg = diff_sum / proves if proves > 0 else 0.0
        born = "✓" if p.get("born_from_reflection") else " "
        print(
            f"  {p.get('name', '?'):<40s} {born:<6s} "
            f"{h.interestingness:>6.3f} {attempts:>4d} {admits:>4d} {proves:>4d} "
            f"{nov_avg:>5.2f} {diff_avg:>5.2f}"
        )


def heuristic_births(events: list[dict]) -> None:
    births = [e for e in events if e["kind"] == "heuristic_birth"]
    if not births:
        return
    print()
    _hr("=")
    print(f"HEURISTIC BIRTHS  ({len(births)})")
    _hr("=")
    for e in births:
        p = e["payload"]
        print(f"  {e['timestamp'][11:19]}  {p.get('name','?'):<40s}  parent={p.get('parent_heuristic','?')}")


def soundness_verdicts(events: list[dict]) -> None:
    sc = [e for e in events if e["kind"] == "heuristic_soundness_check"]
    if not sc:
        return
    print()
    _hr("=")
    print(f"SOUNDNESS GATE VERDICTS  ({len(sc)} total)")
    _hr("=")
    by_verdict = Counter(e["payload"].get("verdict", "?") for e in sc)
    for v, n in by_verdict.most_common():
        passed = "PASS" if v == "passed" else "FAIL"
        print(f"  {passed:<5s}  {v:<35s}  {n:>4d}")


def refutations(data: dict, events: list[dict]) -> None:
    """All refuted conjectures with their witness."""
    objects = data["objects"]
    refuted = [
        (path, o) for path, o in objects.items()
        if o.type == "concept" and o.properties.get("refuted")
    ]
    if not refuted:
        return
    print()
    _hr("=")
    print(f"REFUTATIONS  ({len(refuted)})")
    _hr("=")
    for path, o in refuted:
        p = o.properties
        witness = (p.get("counterexample_witness", "") or "").replace("\n", " | ")
        print(f"  {p.get('name','?'):<40s}  {witness[:200]}")


def proved_theorems_with_provenance(data: dict) -> tuple[int, int]:
    """List all proved theorems, attributing each to its originating heuristic.

    Returns (total_proved, born_attributed_count). The latter is the
    Phase 2 acceptance signal: > 0 means at least one reflection-born
    heuristic contributed a proved theorem.
    """
    objects = data["objects"]
    heuristics_by_name = {
        o.properties.get("name", ""): o
        for o in objects.values() if o.type == "heuristic"
    }
    proved = [
        o for o in objects.values()
        if o.type == "concept" and o.properties.get("kind") == "theorem"
    ]
    proved.sort(key=lambda o: o.properties.get("name", ""))

    total = len(proved)
    born_count = 0

    print()
    _hr("=")
    print(f"PROVED THEOREMS WITH PROVENANCE  ({total})")
    _hr("=")
    print(f"  {'theorem':<40s} {'origin heuristic':<35s} {'born?':<6s}")
    print("  " + "-" * 81)

    for c in proved:
        name = c.properties.get("name", "?")
        origin = c.properties.get("origin_heuristic") or "(none)"
        h_obj = heuristics_by_name.get(origin)
        is_born = bool(h_obj and h_obj.properties.get("born_from_reflection"))
        if is_born:
            born_count += 1
        marker = "✓" if is_born else " "
        print(f"  {name:<40s} {origin:<35s} {marker:<6s}")

    return total, born_count


def trace_kinds(events: list[dict]) -> None:
    if not events:
        return
    print()
    _hr("=")
    print(f"TRACE EVENT BREAKDOWN  ({len(events)} total)")
    _hr("=")
    by_kind = Counter(e["kind"] for e in events)
    for k, n in by_kind.most_common():
        print(f"  {k:<30s} {n:>5d}")


def proof_attempt_outcomes(events: list[dict]) -> None:
    pa = [e for e in events if e["kind"] == "proof_attempt"]
    if not pa:
        return
    print()
    _hr("=")
    print(f"PROOF ATTEMPTS  ({len(pa)})")
    _hr("=")
    by_strategy = Counter()
    successes_by_strategy = Counter()
    for e in pa:
        p = e["payload"]
        s = p.get("strategy", "?")
        by_strategy[s] += 1
        if p.get("outcome") == "success":
            successes_by_strategy[s] += 1
    for s, n in sorted(by_strategy.items()):
        succ = successes_by_strategy[s]
        rate = (succ / n * 100) if n > 0 else 0
        print(f"  {s:<30s}  {succ:>3d} / {n:>3d}  ({rate:>5.1f}%)")


def main() -> int:
    p = argparse.ArgumentParser(prog="cli.summarize", description=__doc__)
    p.add_argument("pickle", help="path to agenda pickle file")
    p.add_argument(
        "--jsonl", default=None,
        help="path to trace.jsonl (default: <pickle-stem>.trace.jsonl alongside)",
    )
    args = p.parse_args()

    pkl_path = _pkl_path(args.pickle)
    data = _load_pkl(pkl_path)

    jsonl_path = Path(args.jsonl) if args.jsonl else _jsonl_for(pkl_path)
    events = _load_jsonl(jsonl_path) if jsonl_path else []

    print()
    print(f"  pickle: {pkl_path}")
    if jsonl_path:
        print(f"  trace:  {jsonl_path}  ({len(events)} events)")
    else:
        print("  trace:  (none — pass --jsonl <path> for richer summary)")

    overview(data)
    heuristic_table(data)
    if events:
        trace_kinds(events)
        heuristic_births(events)
        soundness_verdicts(events)
        proof_attempt_outcomes(events)
    refutations(data, events)
    total_proved, born_attributed = proved_theorems_with_provenance(data)

    # Phase 2 acceptance verdict
    print()
    _hr("=")
    print("PHASE 2 ACCEPTANCE CRITERION")
    _hr("=")
    if born_attributed > 0:
        print(f"  ✓ CLOSED: {born_attributed} of {total_proved} proved theorems attributed to reflection-born heuristics.")
    else:
        n_born = sum(
            1 for o in data["objects"].values()
            if o.type == "heuristic" and o.properties.get("born_from_reflection")
        )
        if n_born == 0:
            print(f"  ✗ no reflection-born heuristics yet ({total_proved} proves all from seed heuristics).")
        else:
            print(
                f"  ⏳ {n_born} reflection-born heuristics admitted, none have contributed a proved theorem yet "
                f"({total_proved} proves all from seed heuristics)."
            )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
