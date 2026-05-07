"""Cross-reference alignment results against proved theorems.

The alignment probe answers "which invented concepts are α-equivalent to
canonical concepts?". The grounding tool answers the downstream question:

    Of the proved theorems whose statement uses invented concepts, how many
    of those invented concepts ALIAS back to canonical (seed) concepts?

A proved theorem stated in invented vocabulary that aliases entirely to seed
vocabulary is a *restatement* (correct, but the content is in Mathlib /
seed terms). A proved theorem stated in invented vocabulary that contains
≥1 genuinely-novel invented concept is *novel content*.

This is the bookkeeping that turns the alignment ratio into a claim about
the discovery corpus.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TheoremGrounding:
    theorem_name: str
    invented_concepts_used: list[str] = field(default_factory=list)
    aliased_count: int = 0  # how many of those alias to a canonical concept
    novel_invented: list[str] = field(default_factory=list)  # the unaliased ones
    statement: str = ""

    @property
    def category(self) -> str:
        if not self.invented_concepts_used:
            return "seed_only"
        if not self.novel_invented:
            return "fully_aliased"  # every invented concept used has a canonical alias
        return "partially_novel"  # ≥1 invented concept that didn't align


def _load_alignment(alignment_json: Path) -> dict[str, str]:
    """Return {invented_name: category}, where category ∈ {alias, synonyms, novel, ...}."""
    blob = json.loads(alignment_json.read_text())
    return {entry["invented_name"]: entry["category"] for entry in blob}


def _extract_used_invented(stmt: str, invented_names: set[str]) -> list[str]:
    """Return invented concept names that appear in the statement.

    Uses a word-boundary match against each invented name.
    """
    used = []
    for nm in invented_names:
        # Look for the name as a separated identifier
        if re.search(rf'\b{re.escape(nm)}\b', stmt):
            used.append(nm)
    return used


def ground_theorems(
    pickle_path: Path,
    alignment_json: Path,
    domain: str,
    seed_names: set[str],
) -> list[TheoremGrounding]:
    """For each proved theorem in the pickle, classify whether its invented
    vocabulary is aliased back to seeds or contains genuinely novel concepts.
    """
    alignment = _load_alignment(alignment_json)
    # invented = anything in the alignment that wasn't classified as alias/synonyms
    aliased = {nm for nm, cat in alignment.items() if cat in ("alias", "synonyms")}
    novel_set = {nm for nm, cat in alignment.items() if cat == "novel"}
    all_invented = set(alignment.keys())

    with pickle_path.open("rb") as f:
        d = pickle.load(f)

    out: list[TheoremGrounding] = []
    for o in d["objects"].values():
        if o.type != "concept":
            continue
        props = o.properties
        if props.get("kind") != "theorem":
            continue
        if props.get("domain") != domain:
            continue
        proof = props.get("lean_proof") or ""
        if not proof or "sorry" in proof:
            continue  # unproven
        stmt = props.get("lean_statement", "")
        used = _extract_used_invented(stmt, all_invented)
        novel = [n for n in used if n in novel_set or n not in aliased]
        out.append(TheoremGrounding(
            theorem_name=props.get("name", o.path),
            invented_concepts_used=used,
            aliased_count=sum(1 for n in used if n in aliased),
            novel_invented=novel,
            statement=stmt[:300],
        ))
    return out


def write_grounding_report(groundings: list[TheoremGrounding], out_path: Path) -> None:
    from collections import Counter
    cats = Counter(g.category for g in groundings)
    md = ["# Grounding report", ""]
    md.append(f"Total proved theorems: **{len(groundings)}**")
    for cat in ("seed_only", "fully_aliased", "partially_novel"):
        n = cats.get(cat, 0)
        md.append(f"- **{cat}**: {n}")
    md.append("")

    md.append("## Theorems with novel-content concepts (`partially_novel`)")
    novels = [g for g in groundings if g.category == "partially_novel"]
    if not novels:
        md.append("_None._")
    for g in novels:
        md.append(f"### `{g.theorem_name}`")
        md.append(f"Novel invented concepts: {', '.join(f'`{n}`' for n in g.novel_invented)}")
        md.append(f"Other invented (aliased): "
                  f"{', '.join(f'`{n}`' for n in g.invented_concepts_used if n not in g.novel_invented) or '_none_'}")
        md.append(f"```lean\n{g.statement}\n```")
        md.append("")

    md.append("## Theorems whose invented vocabulary is fully aliased back to seed (`fully_aliased`)")
    aliased = [g for g in groundings if g.category == "fully_aliased"]
    if not aliased:
        md.append("_None._")
    for g in aliased:
        aliases = ", ".join(f"`{n}`" for n in g.invented_concepts_used)
        md.append(f"- `{g.theorem_name}` — uses invented {aliases} (all alias back to seed)")
    md.append("")

    md.append("## Theorems stated in seed vocabulary only (`seed_only`)")
    seeds = [g for g in groundings if g.category == "seed_only"]
    md.append(f"_({len(seeds)} theorems — listed by name only)_")
    md.append("")
    for g in seeds:
        md.append(f"- `{g.theorem_name}`")

    out_path.write_text("\n".join(md))
