"""Definitional alignment probe.

For each "invented" concept produced by a discovery run, attempt to prove
that it is iff-equivalent to a "canonical" concept (a seeded definition,
or — Phase 2 — a Mathlib predicate). The Lean kernel acts as the
equivalence oracle: if the iff theorem proves, the two definitions are
α-equivalent under unfolding + the chosen tactic.

Intended use:

    align_run.py --pickle agenda-discovery-matroid.pkl
                 --domain matroid_theory
                 --output outputs/align-matroid.json

Output: each invented concept is tagged as one of
  - "alias": iff to a single canonical concept (the one it matches)
  - "synonyms": iff to multiple canonical concepts (and they're mutually iff
                 — which is fine, just means the system invented a mid-cluster name)
  - "novel": no iff proves; this is candidate genuinely-new content
  - "shape_mismatch": no canonical concept has compatible argument shape
  - "unparseable": couldn't parse the lean_statement to extract a signature

Phase 1 only compares against seed concepts. Phase 2 (not implemented in
this initial pass) would extend the candidate pool to a curated list of
Mathlib predicates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Match a top-level `def NAME {implicits} (explicit_args) : RETTYPE := BODY`.
# The args group matches a sequence of {…}/(…)/[…] groups (so the `:` inside
# binders like `{α : Type*}` doesn't bleed into the rettype slot).
_DEF_RE = re.compile(
    r"""^\s*(?:noncomputable\s+|partial\s+)?
        def\s+(?P<name>[A-Za-z_][A-Za-z0-9_']*)\s*
        (?P<args>(?:\s*(?:\{[^}]*\}|\([^)]*\)|\[[^\]]*\]))*)\s*
        :\s*(?P<rettype>[^:=]+?)\s*
        :=\s*(?P<body>.*)$""",
    re.VERBOSE | re.DOTALL,
)

# `abbrev` form (used in modarith/group_theory seeds for some predicates)
_ABBREV_RE = re.compile(
    r"""^\s*abbrev\s+(?P<name>[A-Za-z_][A-Za-z0-9_']*)\s*
        (?P<args>(?:\s*(?:\{[^}]*\}|\([^)]*\)|\[[^\]]*\]))*)\s*
        :\s*(?P<rettype>[^:=]+?)\s*
        :=\s*(?P<body>.*)$""",
    re.VERBOSE | re.DOTALL,
)

# Parse the args string into [(kind, name, type)] tuples, where kind is
# 'implicit' for `{x : T}` and 'explicit' for `(x : T)`. Instance args
# `[Inst T]` are passed through as kind='instance'.
_ARG_RE = re.compile(
    r"""(?:
            \{\s*(?P<imp_names>[^:}]+?)\s*:\s*(?P<imp_type>[^}]+?)\s*\}
          | \(\s*(?P<exp_names>[^:)]+?)\s*:\s*(?P<exp_type>[^)]+?)\s*\)
          | \[\s*(?P<inst_body>[^\]]+?)\s*\]
        )""",
    re.VERBOSE,
)


@dataclass
class ParsedDef:
    name: str
    implicits: list[tuple[str, str]] = field(default_factory=list)
    explicits: list[tuple[str, str]] = field(default_factory=list)
    instances: list[str] = field(default_factory=list)
    rettype: str = ""
    body: str = ""

    @property
    def signature_key(self) -> str:
        """A normalized key matching the explicit-argument types only.

        Used to bucket definitions that *could* be iff-compared. Implicit
        type parameters and instance args are excluded from the key (they
        get α-renamed during probe rendering).
        """
        return "|".join(t.strip() for _, t in self.explicits) + f"->{self.rettype.strip()}"


def parse_definition(lean_statement: str) -> Optional[ParsedDef]:
    """Parse a `def` or `abbrev` statement into ParsedDef. Returns None if unparseable."""
    if not lean_statement:
        return None

    src = lean_statement.strip()
    m = _DEF_RE.match(src) or _ABBREV_RE.match(src)
    if not m:
        return None

    parsed = ParsedDef(name=m.group("name"), rettype=m.group("rettype"), body=m.group("body").strip())

    args_str = m.group("args")
    for am in _ARG_RE.finditer(args_str):
        if am.group("imp_names"):
            for nm in am.group("imp_names").split():
                parsed.implicits.append((nm, am.group("imp_type")))
        elif am.group("exp_names"):
            for nm in am.group("exp_names").split():
                parsed.explicits.append((nm, am.group("exp_type")))
        elif am.group("inst_body"):
            parsed.instances.append(am.group("inst_body"))

    return parsed


# ---------------------------------------------------------------------------
# Probe rendering
# ---------------------------------------------------------------------------

def _renamed_args(parsed: ParsedDef, target_name: str) -> tuple[str, str]:
    """Render the canonical (re-namespaced) argument list and call-site for the iff theorem.

    Returns (binders, call_args) where binders is the function-style argument
    list for the theorem and call_args is the comma-separated argument list
    for invocation.
    """
    binders = []
    call = []
    # implicits
    for nm, typ in parsed.implicits:
        binders.append(f"{{{nm} : {typ}}}")
    # instances (passed through verbatim — they bind the implicit type)
    for inst in parsed.instances:
        binders.append(f"[{inst}]")
    # explicits
    for nm, typ in parsed.explicits:
        binders.append(f"({nm} : {typ})")
        call.append(nm)
    return " ".join(binders), " ".join(call)


# Tactic ladder, ordered by increasing effort. The first that proves wins.
# Kept short on purpose — empirically `unfold; aesop` catches almost every
# case the others would catch, and Mathlib import dominates per-probe cost
# anyway.
TACTIC_LADDER: list[tuple[str, str]] = [
    ("unfold_rfl", "unfold {names}; rfl"),
    ("unfold_tauto", "unfold {names}; tauto"),
    ("unfold_aesop", "unfold {names}; aesop"),
]


def render_iff_probe(
    invented: ParsedDef,
    canonical: ParsedDef,
    invented_stmt: str,
    canonical_stmt: str,
    tactic: str,
) -> str:
    """Render a Lean file that defines both predicates and asserts their iff.

    Returns the full Lean source, ready to feed to the LeanBackend verifier.
    """
    binders, call = _renamed_args(invented, "_")
    names_for_unfold = f"{invented.name} {canonical.name}"
    names_csv = f"{invented.name}, {canonical.name}"

    # Substitute placeholders in the tactic template
    rendered_tac = tactic.format(names=names_for_unfold, names_csv=names_csv)

    return f"""import Mathlib

set_option maxHeartbeats 400000

{invented_stmt}

{canonical_stmt}

theorem _align_{invented.name}_{canonical.name} {binders} :
    {invented.name} {call} ↔ {canonical.name} {call} := by
  {rendered_tac}
"""


# ---------------------------------------------------------------------------
# Alignment results
# ---------------------------------------------------------------------------


@dataclass
class AlignmentMatch:
    canonical_name: str
    tactic_name: str
    tactic_str: str
    log_excerpt: str = ""


@dataclass
class AlignmentResult:
    invented_name: str
    invented_signature: str
    matches: list[AlignmentMatch] = field(default_factory=list)
    note: str = ""  # for shape_mismatch, unparseable, etc.

    @property
    def category(self) -> str:
        if self.note:
            return self.note
        if not self.matches:
            return "novel"
        if len(self.matches) == 1:
            return "alias"
        return "synonyms"
