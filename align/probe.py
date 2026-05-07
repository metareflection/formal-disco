"""Run alignment probes against a canonical concept set."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from language import Language, Program, VerificationOutcome

from . import (
    AlignmentMatch,
    AlignmentResult,
    ParsedDef,
    TACTIC_LADDER,
    parse_definition,
    render_iff_probe,
)


logger = logging.getLogger(__name__)


def _try_one_pair(
    backend,
    invented: ParsedDef,
    invented_stmt: str,
    canonical: ParsedDef,
    canonical_stmt: str,
    timeout: float,
) -> AlignmentMatch | None:
    """Try the tactic ladder against this (invented, canonical) pair.

    Returns the first matching tactic, or None if none succeed.
    """
    for tactic_name, tactic_str in TACTIC_LADDER:
        src = render_iff_probe(
            invented=invented,
            canonical=canonical,
            invented_stmt=invented_stmt,
            canonical_stmt=canonical_stmt,
            tactic=tactic_str,
        )
        prog = Program(src, Language.LEAN, name=f"_align_{invented.name}_{canonical.name}_{tactic_name}.lean")
        try:
            ver = backend.verify(prog, timeout=timeout)
        except Exception as e:
            logger.debug(f"verify error on {invented.name}↔{canonical.name} ({tactic_name}): {e}")
            continue
        if ver.outcome == VerificationOutcome.SUCCESS:
            log_excerpt = ((ver.stdout or "") + (ver.stderr or ""))[-200:]
            return AlignmentMatch(
                canonical_name=canonical.name,
                tactic_name=tactic_name,
                tactic_str=tactic_str,
                log_excerpt=log_excerpt,
            )
    return None


def align_one(
    backend,
    invented_concept: dict,
    canonical_concepts: list[dict],
    timeout: float = 60.0,
    *,
    symmetric: bool = False,
) -> AlignmentResult:
    """Align a single invented concept against all compatible canonical concepts.

    invented_concept and canonical_concepts are dicts with keys
    'name' and 'lean_statement'.

    If `symmetric=True`, skip candidates with names <= the invented name.
    Use in invented-vs-invented mode so each pair {A, B} is checked once.
    """
    invented_parsed = parse_definition(invented_concept.get("lean_statement", ""))
    if invented_parsed is None:
        return AlignmentResult(
            invented_name=invented_concept.get("name", "?"),
            invented_signature="",
            note="unparseable",
        )

    sig = invented_parsed.signature_key
    result = AlignmentResult(invented_name=invented_parsed.name, invented_signature=sig)

    # Filter canonical pool to compatible signatures
    compatible: list[tuple[ParsedDef, str]] = []
    for c in canonical_concepts:
        cp = parse_definition(c.get("lean_statement", ""))
        if cp is None:
            continue
        if cp.signature_key != sig:
            continue
        if cp.name == invented_parsed.name:
            continue  # don't compare against itself
        if symmetric and cp.name <= invented_parsed.name:
            continue  # let the partner half do this pair
        compatible.append((cp, c.get("lean_statement", "")))

    if not compatible:
        result.note = "shape_mismatch"
        return result

    invented_stmt = invented_concept.get("lean_statement", "")
    for canonical_parsed, canonical_stmt in compatible:
        match = _try_one_pair(
            backend, invented_parsed, invented_stmt,
            canonical_parsed, canonical_stmt, timeout
        )
        if match:
            result.matches.append(match)

    return result


def align_many(
    backend,
    invented_concepts: list[dict],
    canonical_concepts: list[dict],
    *,
    max_workers: int = 6,
    timeout: float = 60.0,
    symmetric: bool = False,
) -> list[AlignmentResult]:
    """Run align_one on each invented concept in parallel."""
    results: list[AlignmentResult | None] = [None] * len(invented_concepts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(align_one, backend, ic, canonical_concepts, timeout, symmetric=symmetric): i
            for i, ic in enumerate(invented_concepts)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                logger.error(f"align_one crashed on {invented_concepts[i].get('name')}: {e}")
                results[i] = AlignmentResult(
                    invented_name=invented_concepts[i].get("name", "?"),
                    invented_signature="",
                    note="error",
                )
    return [r for r in results if r is not None]
