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
from align.cluster import cluster_synonyms


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
    p.add_argument("--mode", choices=("vs-seed", "invented-vs-invented"), default="vs-seed",
                   help="vs-seed: align each invented concept to seed concepts. "
                        "invented-vs-invented: pairwise within invented pool to find synonym clusters.")
    p.add_argument("--canonical", type=Path, default=None,
                   help="In vs-seed mode, use this JSON file (same schema as --seed) "
                        "as the canonical pool instead of --seed. "
                        "Use to align against Mathlib wrappers (data/canonical_matroid_mathlib.json).")
    args = p.parse_args()

    seed_concepts = load_seed_concepts(args.seed)
    seed_names = {c["name"] for c in seed_concepts}
    invented = load_invented_concepts(args.pickle, seed_names, args.domain)
    logger.info(f"Loaded {len(seed_concepts)} seed concepts, {len(invented)} invented concepts")

    canonical_concepts = seed_concepts
    if args.canonical:
        canonical_concepts = load_seed_concepts(args.canonical)
        logger.info(f"Using --canonical pool from {args.canonical}: {len(canonical_concepts)} concepts")

    if args.limit:
        invented = invented[: args.limit]
        logger.info(f"Limiting to first {len(invented)} for this run")

    # Lazy: import language only when needed (so --help works without LEAN_PROJECT_DIR)
    from language import Language
    backend = Language.LEAN.get_backend()

    t0 = time.time()
    if args.mode == "vs-seed":
        canonical = canonical_concepts
    else:
        # invented-vs-invented: each concept's canonical pool is the rest of the invented set.
        # align_one already excludes self-comparisons by name.
        canonical = invented

    results = align_many(backend, invented, canonical, max_workers=args.workers,
                         timeout=args.timeout, symmetric=(args.mode == "invented-vs-invented"))
    dt = time.time() - t0
    logger.info(f"Aligned {len(results)} concepts in {dt:.1f}s (mode={args.mode})")

    write_report(results, args.output)
    logger.info(f"Wrote {args.output}/alignment.{{json,md}}")

    if args.mode == "invented-vs-invented":
        clusters = cluster_synonyms(results)
        cluster_md = ["# Synonym clusters", ""]
        cluster_md.append(f"Found **{len(clusters)}** non-trivial clusters "
                          f"(size ≥ 2) in {len(results)} invented concepts.\n")
        for c in clusters:
            cluster_md.append(f"## Cluster (size {c.size}, rep `{c.representative}`)")
            for m in c.members:
                cluster_md.append(f"- `{m}`")
            cluster_md.append("")
            cluster_md.append("Proven edges:")
            for a, b, tac in c.edges:
                cluster_md.append(f"- `{a}` ↔ `{b}` (via `{tac}`)")
            cluster_md.append("")
        (args.output / "synonyms.md").write_text("\n".join(cluster_md))

        # JSON for downstream tools
        cluster_json = [
            {"representative": c.representative, "members": c.members,
             "edges": [{"a": a, "b": b, "tactic": t} for a, b, t in c.edges]}
            for c in clusters
        ]
        (args.output / "synonyms.json").write_text(json.dumps(cluster_json, indent=2))
        logger.info(f"Wrote {args.output}/synonyms.{{json,md}} ({len(clusters)} clusters)")

    cats = Counter(r.category for r in results)
    print("=" * 60)
    for cat, n in cats.most_common():
        print(f"  {cat}: {n}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
