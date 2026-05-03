"""Mechanical soundness gate for proposed-new heuristics.

When ``ReflectionWorker`` proposes a new heuristic (Phase 2 of
PROPOSAL_EURISKO.md), this check decides whether to admit it. The
proposer/checker asymmetry from the proposal:

  - The LLM proposes creatively (untrusted).
  - The check is mechanical and trusted.

What this gate enforces (Phase 1.5 stub — three cheap tests, no live LLM):

  1. **Template plausibility**: prompt template is present and not trivially
     short. Catches malformed reflection outputs.

  2. **applies_to non-empty**: the heuristic's structural filter matches at
     least one concept in a held-out harness. A heuristic that applies to
     nothing is dead on arrival.

  3. **Prompt-similarity dedup**: the new heuristic's prompt isn't a near-
     paraphrase (Jaccard token similarity) of an existing one. Stops the
     system from minting trivial restatements of "specialize" / "generalize"
     under different names.

The "expensive" parts of the proposal — running the proposed heuristic's
``propose`` step on the harness and counting typecheck-pass rate — are
intentionally NOT in this stub. They require a live LLM call per heuristic
birth, and we want the structural gate landing first so we can validate the
slot redesign end-to-end without burning API quota on every reflection event.

Phase 2 follow-up will add ``check_heuristic_pass_rate`` here with the LLM
harness; the result will multiply into the overall verdict.
"""

from __future__ import annotations

import re
from typing import Iterable

from agenda import Object
from discovery.checks import CheckResult
from discovery.heuristic import Heuristic


def _tokens(text: str) -> set[str]:
    """Lowercased word-token set, for Jaccard similarity."""
    return set(re.findall(r"\w+", text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Symmetric Jaccard over word-token sets in [0, 1]."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _existing_template(obj: Object) -> str:
    return (obj.content or b"").decode("utf-8", errors="replace")


def check_heuristic_soundness(
    *,
    proposed: Heuristic,
    harness: list[Object],
    existing_heuristics: list[Object],
    min_template_length: int = 30,
    min_applies: int = 1,
    similarity_threshold: float = 0.7,
) -> CheckResult:
    """Run the soundness pipeline. Returns CheckResult with structured verdict.

    ``harness`` is a list of concept Objects (held-out, ideally not the ones
    that prompted the reflection event). ``existing_heuristics`` is the
    current heuristic pool for similarity dedup.

    Return verdicts:
      - ``"passed"``: all gates cleared
      - ``"empty_template"``: template too short / missing
      - ``"applies_to_empty"``: no harness concepts match
      - ``"too_similar_to_existing"``: prompt is a near-paraphrase
    """
    # 1. Template plausibility (only for prompt-template propose).
    if proposed.propose.kind == "prompt_template":
        tmpl = proposed.propose.template.strip()
        if len(tmpl) < min_template_length:
            return CheckResult(
                passed=False,
                verdict="empty_template",
                reason=f"template length {len(tmpl)} < {min_template_length}",
                details={"length": len(tmpl)},
            )

    # 2. applies_to non-empty over harness.
    if harness:
        matches = sum(1 for c in harness if proposed.applies_to.matches(c))
    else:
        matches = 0
    if matches < min_applies:
        return CheckResult(
            passed=False,
            verdict="applies_to_empty",
            reason=f"matches {matches} / {len(harness)} harness concepts",
            details={"n_applicable": matches, "harness_size": len(harness)},
        )

    # 3. Prompt-similarity dedup vs existing heuristics.
    if proposed.propose.kind == "prompt_template" and existing_heuristics:
        best_sim = 0.0
        best_label = ""
        for h_obj in existing_heuristics:
            sim = jaccard_similarity(proposed.propose.template, _existing_template(h_obj))
            if sim > best_sim:
                best_sim = sim
                best_label = h_obj.properties.get("name", h_obj.path)
        if best_sim >= similarity_threshold:
            return CheckResult(
                passed=False,
                verdict="too_similar_to_existing",
                reason=f"jaccard={best_sim:.3f} to {best_label}",
                witness=best_label,
                details={"similarity": best_sim, "nearest": best_label},
            )

    return CheckResult(
        passed=True,
        verdict="passed",
        reason=f"applies_to {matches}/{len(harness)} harness concepts",
        details={"n_applicable": matches, "harness_size": len(harness)},
    )


def collect_harness(agenda, *, exclude_paths: Iterable[str] = (), max_size: int = 30) -> list[Object]:
    """Sample a held-out concept harness from the agenda.

    Uses the private ``_objects`` dict (same Phase 1 expedient as
    ``checks/novelty.py``). Excludes any concept paths in ``exclude_paths``
    — typically the concept that triggered the reflection event, so the
    soundness check doesn't get a free pass by matching its own input.
    """
    objects = getattr(agenda, "_objects", None)
    if not objects:
        return []
    excl = set(exclude_paths)
    out: list[Object] = []
    for path, obj in objects.items():
        if obj.type != "concept":
            continue
        if path in excl:
            continue
        out.append(obj)
        if len(out) >= max_size:
            break
    return out


def collect_existing_heuristics(agenda) -> list[Object]:
    """All heuristic objects currently in the agenda (for similarity dedup)."""
    objects = getattr(agenda, "_objects", None)
    if not objects:
        return []
    return [obj for obj in objects.values() if obj.type == "heuristic"]
