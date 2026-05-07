#!/usr/bin/env python3
"""Run a definitional alignment probe over a discovery-run pickle.

For each invented concept (kind ∈ {definition, operation}, name not in the
seed set), attempts to prove iff with each compatible seed concept via the
Lean kernel. Outputs JSON + Markdown report.

Usage:
    eval $(opam env)  # not needed for Lean, but doesn't hurt
    LEAN_PROJECT_DIR=../LeanDisco python align_run.py \\
        --pickle agenda-discovery-matroid.pkl \\
        --seed data/seed_matroid_theory.json \\
        --domain matroid_theory \\
        --output outputs/align-matroid

Phase 1 only: invented-vs-seed. Phase 2 (Mathlib alignment) deferred.
"""

import argparse
import json
import logging
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

from align.probe import align_many


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_seed_concepts(seed_path: Path) -> list[dict]:
    with seed_path.open() as f:
        return [c for c in json.load(f) if c.get("kind") in ("definition", "operation")]


def load_invented_concepts(pickle_path: Path, seed_names: set[str], domain: str) -> list[dict]:
    with pickle_path.open("rb") as f:
        d = pickle.load(f)
    out = []
    for o in d["objects"].values():
        if o.type != "concept":
            continue
        props = o.properties
        if props.get("domain") != domain:
            continue
        if props.get("kind") not in ("definition", "operation"):
            continue
        nm = props.get("name", "")
        if not nm or nm in seed_names:
            continue
        out.append({
            "name": nm,
            "lean_statement": props.get("lean_statement", ""),
            "kind": props.get("kind"),
            "description": props.get("description", ""),
        })
    return out


def write_report(results: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON: full machine-readable
    json_blob = []
    for r in results:
        json_blob.append({
            "invented_name": r.invented_name,
            "invented_signature": r.invented_signature,
            "category": r.category,
            "note": r.note,
            "matches": [
                {
                    "canonical_name": m.canonical_name,
                    "tactic_name": m.tactic_name,
                    "tactic_str": m.tactic_str,
                }
                for m in r.matches
            ],
        })
    (out_dir / "alignment.json").write_text(json.dumps(json_blob, indent=2))

    # Markdown: human-readable
    md = ["# Alignment probe report", ""]
    cat_counts = Counter(r.category for r in results)
    md.append(f"Total invented concepts: **{len(results)}**\n")
    for cat in ("alias", "synonyms", "novel", "shape_mismatch", "unparseable", "error"):
        if cat_counts.get(cat, 0):
            md.append(f"- **{cat}**: {cat_counts[cat]}")
    md.append("")
    # Categorized listings
    for cat in ("alias", "synonyms", "novel", "shape_mismatch", "unparseable", "error"):
        rs = [r for r in results if r.category == cat]
        if not rs:
            continue
        md.append(f"## {cat.title()} ({len(rs)})")
        for r in rs:
            line = f"- `{r.invented_name}`"
            if r.matches:
                tags = ", ".join(f"{m.canonical_name}({m.tactic_name})" for m in r.matches)
                line += f" → {tags}"
            md.append(line)
        md.append("")
    (out_dir / "alignment.md").write_text("\n".join(md))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pickle", required=True, type=Path)
    p.add_argument("--seed", required=True, type=Path)
    p.add_argument("--domain", required=True)
    p.add_argument("--output", required=True, type=Path,
                   help="Directory to write alignment.json + alignment.md")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--timeout", type=float, default=60.0,
                   help="Per-probe Lean timeout (seconds)")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only process this many invented concepts (for dry-runs)")
    args = p.parse_args()

    seed_concepts = load_seed_concepts(args.seed)
    seed_names = {c["name"] for c in seed_concepts}
    invented = load_invented_concepts(args.pickle, seed_names, args.domain)
    logger.info(f"Loaded {len(seed_concepts)} seed concepts, {len(invented)} invented concepts")

    if args.limit:
        invented = invented[: args.limit]
        logger.info(f"Limiting to first {len(invented)} for this run")

    # Lazy: import language only when needed (so --help works without LEAN_PROJECT_DIR)
    from language import Language
    backend = Language.LEAN.get_backend()

    t0 = time.time()
    results = align_many(backend, invented, seed_concepts, max_workers=args.workers, timeout=args.timeout)
    dt = time.time() - t0
    logger.info(f"Aligned {len(results)} concepts in {dt:.1f}s")

    write_report(results, args.output)
    logger.info(f"Wrote {args.output}/alignment.{{json,md}}")

    cats = Counter(r.category for r in results)
    print("=" * 60)
    for cat, n in cats.most_common():
        print(f"  {cat}: {n}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
