"""Slotted view over a heuristic agenda Object.

Phase 2 of PROPOSAL_EURISKO.md: reintroduce eurisclo-style slot structure on
heuristics so reflection-born heuristics can carry custom propose/check/commit
behavior while staying serializable in the agenda's pickle.

Design notes
------------

The 9 seed heuristics in v2 are *prompt-template* heuristics: they hold a
single ``content`` blob (the LLM prompt) and rely on workers to interpret it.
That collapse means heuristic-birth is structurally meaningless — a "new
heuristic" is just a new prompt string with no behavioral difference from the
old ones (cf. PROPOSAL_EURISKO.md §1).

The fix here is a *runtime wrapper*: ``Heuristic`` is constructed from an
existing agenda Object via ``Heuristic.from_object``, so the storage layout
doesn't change and existing heuristics keep working. The slot specs
(``AppliesToSpec`` / ``ProposeSpec`` / ``CheckSpec`` / ``CommitSpec``) are
deliberately *structural* rather than callable — they're dicts of
declarative parameters, not raw Python functions, so they can be pickled and
inspected without the security/version-skew foot-guns that come with
shipping callables in checkpoint files.

The "default" kinds reproduce v2 behavior exactly:

  - ``AppliesToSpec``: filter concepts by ``input_concept_kinds`` ∩ ``input_tags``
  - ``ProposeSpec(kind="prompt_template")``: LLM call with the heuristic's
    prompt template (the heuristic's ``content`` blob)
  - ``CheckSpec(kind="default_pipeline")``: typecheck + novelty + counterexample
    (for conjectures), as DiscoveryWorker already does
  - ``CommitSpec(kind="default")``: create concept Object + queue prove/discover

Phase 2 reflection-born heuristics can carry alternate kinds (``"code"`` for
propose, ``"no_counterexample"`` for check, etc.) and the workers will route
on the kind tag. None of those alternates exist yet — this is the seam for
future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agenda import Object


@dataclass
class AppliesToSpec:
    """Cheap structural filter for which concepts a heuristic applies to.

    Mirrors eurisclo's ``if-potentially-relevant``: a quick predicate before
    the more expensive worth-sort step.
    """

    concept_kinds: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def matches(self, concept_obj: Object) -> bool:
        """Return True if the heuristic should consider this concept."""
        if self.concept_kinds:
            kind = concept_obj.properties.get("kind", "")
            if kind not in self.concept_kinds:
                return False
        if self.tags:
            concept_tags = set(concept_obj.properties.get("tags", []))
            if not concept_tags & set(self.tags):
                return False
        return True


@dataclass
class ProposeSpec:
    """How a heuristic generates new candidate concepts.

    ``kind`` selects the interpreter:
      - ``"prompt_template"``: LLM call with ``template`` as the user-side
        prompt. Default for v2 seed heuristics.
      - ``"code"``: not yet implemented; reserved for Phase 2 reflection-born
        heuristics that ship a code-level mutation function.
    """

    kind: str = "prompt_template"
    template: str = ""


@dataclass
class CheckSpec:
    """Which mechanical checks gate admission of this heuristic's candidates.

    ``kind`` selects the pipeline:
      - ``"default_pipeline"``: typecheck (stub-with-sorry) + novelty + (for
        conjectures) counterexample. The current DiscoveryWorker behavior.
      - ``"no_counterexample"``: skip the slim_check probe (use for abstract
        domains where SampleableExt isn't available).
      - ``"novelty_only"``: skip everything except novelty. Useful for
        heuristics that only generate definitions (not conjectures).

    Other kinds may be introduced as the system gains more checkers.
    """

    kind: str = "default_pipeline"


@dataclass
class CommitSpec:
    """What to do with admitted candidates.

      - ``"default"``: create concept Object + queue prove (for conjectures)
        or discover (for definitions). Matches the v2 DiscoveryWorker behavior.
    """

    kind: str = "default"


@dataclass
class Heuristic:
    """Slotted view over a heuristic agenda Object.

    Constructed via ``Heuristic.from_object(obj)`` from an existing heuristic
    Object. The original Object remains the source of truth; this wrapper just
    interprets its properties + content through the slot specs.

    Worth components (attempts/admits/proves/novelty_sum/difficulty_sum) are
    *not* duplicated here — they live on the underlying Object's properties
    and are read via ``discovery.worth.WorthComponents.from_properties``.
    """

    name: str
    heuristic_kind: str
    applies_to: AppliesToSpec
    propose: ProposeSpec
    check: CheckSpec
    commit: CommitSpec
    born_from_reflection: bool = False

    @classmethod
    def from_object(cls, obj: Object) -> "Heuristic":
        p = obj.properties
        # Existing v2 seed heuristics don't carry propose_kind / check_kind /
        # commit_kind in their properties, so we fall through to the defaults
        # — which reproduce the v2 behavior exactly.
        return cls(
            name=str(p.get("name", "")),
            heuristic_kind=str(p.get("heuristic_kind", "concept")),
            applies_to=AppliesToSpec(
                concept_kinds=list(p.get("input_concept_kinds", []) or []),
                tags=list(p.get("input_tags", []) or []),
            ),
            propose=ProposeSpec(
                kind=str(p.get("propose_kind", "prompt_template")),
                template=(obj.content or b"").decode("utf-8", errors="replace"),
            ),
            check=CheckSpec(
                kind=str(p.get("check_kind", "default_pipeline")),
            ),
            commit=CommitSpec(
                kind=str(p.get("commit_kind", "default")),
            ),
            born_from_reflection=bool(p.get("born_from_reflection", False)),
        )

    def to_object_properties(self) -> dict[str, Any]:
        """Render the spec part of this Heuristic into agenda Object properties.

        Worth components and the prompt template (which lives in ``content``,
        not ``properties``) are NOT included — those are managed elsewhere.
        Use this when materializing a freshly-minted heuristic spec into an
        agenda Object.
        """
        return {
            "name": self.name,
            "heuristic_kind": self.heuristic_kind,
            "input_concept_kinds": list(self.applies_to.concept_kinds),
            "input_tags": list(self.applies_to.tags),
            "propose_kind": self.propose.kind,
            "check_kind": self.check.kind,
            "commit_kind": self.commit.kind,
            "born_from_reflection": self.born_from_reflection,
        }

    def template(self) -> str:
        """Prompt-template propose body, or empty string if a different kind."""
        return self.propose.template if self.propose.kind == "prompt_template" else ""


def matches_concept(heuristic_obj: Object, concept_obj: Object) -> bool:
    """Convenience wrapper: build a Heuristic view and apply its filter.

    Use this when you have a heuristic Object directly and just need a
    yes/no answer. For repeated filtering, build the ``Heuristic`` once.
    """
    return Heuristic.from_object(heuristic_obj).applies_to.matches(concept_obj)
