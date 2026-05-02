"""Mechanical pre-proof checkers.

Each check is a small, deterministic test that admits or rejects a candidate
*before* it reaches an LLM-driven prover. The proposer-checker asymmetry from
PROPOSAL_EURISKO.md lives here: proposing is creative and untrusted (the LLM),
checking is mechanical and trusted (these modules).

A check returns a ``CheckResult`` with a structured verdict and (when refuted)
a witness or reason that downstream code can record in the trace.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CheckResult:
    """Verdict of a mechanical check on a candidate.

    Attributes:
        passed: True if the candidate survives the check (proceed).
        verdict: Short tag describing the outcome — ``"refuted"``,
            ``"passed"``, ``"too_similar"``, ``"inconclusive"``, etc.
            Used for trace records and worth attribution.
        reason: Human-readable explanation, suitable for a trace summary line.
        witness: Counterexample or matched-existing reference, when applicable.
        details: Free-form payload (raw verifier output, distance, etc.).
    """

    passed: bool
    verdict: str
    reason: str = ""
    witness: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
