"""Counterexample search via the ``plausible`` tactic.

Pre-proof gate: turn the conjecture into ``example ... := by plausible`` and
let Lean's property-based tester sample concrete instances. If a counterexample
shows up, we refute *before* spending an Opus call on a doomed proof.

Inconclusive outcomes (plausible can't sample, no Decidable instance, or it
just timed out) are NOT treated as refutation — the conjecture proceeds normally.
The whole point of the asymmetry is that the check only speaks when it has
mechanical evidence; silence means "I don't know," not "false."

Note on naming: this used to be ``slim_check`` in older Mathlib. The Lean FRO
extracted it as a standalone ``Plausible`` package; the API is the same.
"""

from __future__ import annotations

import logging
import re

from discovery import resolve_imports
from discovery.checks import CheckResult
from language import Language, Program, VerificationOutcome
from language.lean import LeanBackend

logger = logging.getLogger(__name__)


def _strip_proof_body(statement: str) -> str:
    """Remove any ``:= ...`` proof body so we can attach our own ``by slim_check``."""
    cleaned = re.sub(r":=\s*by\b.*", "", statement, flags=re.DOTALL).rstrip()
    cleaned = re.sub(r":=\s*(?!sorry).*", "", cleaned, flags=re.DOTALL).rstrip()
    cleaned = re.sub(r":=\s*sorry\s*$", "", cleaned, flags=re.DOTALL).rstrip()
    return cleaned


def _to_example(statement: str) -> str:
    """Convert ``theorem foo ... :`` (or ``lemma`` etc.) to ``example ... :`` so
    we can re-attach a fresh proof body without redeclaring the name."""
    return re.sub(
        r"^\s*(?:theorem|lemma|def|abbrev|instance)\s+\S+",
        "example",
        statement,
        count=1,
    )


def build_probe(*, statement: str, imports: list[str], preamble: str, num_inst: int) -> str:
    """Build a Lean source file that runs plausible on the conjecture."""
    body = _to_example(_strip_proof_body(statement))
    import_lines = ["import Plausible"]
    import_lines += [f"import {imp}" for imp in resolve_imports(imports)]
    parts = [
        "\n".join(import_lines),
        "",
        preamble,
        "",
        f"{body} := by plausible (config := {{ numInst := {num_inst} }})",
        "",
    ]
    return "\n".join(parts)


# plausible writes a banner like:
#   ===================
#   Found a counter-example!
#   n := 1
#   issue: 1 = 0 does not hold
# Older slim_check formats included for forward-compatibility with mixed setups.
_REFUTED_MARKERS = (
    "Found a counter-example!",
    "Found problems!",
    "===Found a counterexample===",
    "Counter-examples found.",
)


def _looks_refuted(output: str) -> bool:
    return any(m in output for m in _REFUTED_MARKERS)


def _extract_witness(output: str) -> str:
    """Pull out the lines that look like the counterexample binding(s).

    plausible typically prints assignments like ``x := 0`` after the marker.
    We grab a bounded slice as the witness so the trace stays small.
    """
    for marker in _REFUTED_MARKERS:
        idx = output.find(marker)
        if idx >= 0:
            tail = output[idx : idx + 800]
            return tail.strip()
    return output[:500].strip()


def check_counterexample(
    *,
    statement: str,
    imports: list[str],
    preamble: str,
    backend: LeanBackend | None = None,
    num_inst: int = 50,
    timeout: float = 60.0,
) -> CheckResult:
    """Try to refute ``statement`` with slim_check. Best-effort.

    Returns a ``CheckResult`` with:
      - ``passed=False, verdict='refuted'`` if slim_check produced a counterexample.
      - ``passed=True, verdict='passed'`` if slim_check ran clean (no counterexample
        found within ``num_inst`` samples and the file compiled).
      - ``passed=True, verdict='inconclusive'`` if slim_check couldn't be applied
        (compile failure for unrelated reasons, no Decidable/Sampleable, etc.).
        Inconclusive lets the conjecture proceed — silence is not refutation.
    """
    if backend is None:
        backend = Language.LEAN.get_backend()

    if not statement:
        return CheckResult(passed=True, verdict="inconclusive", reason="empty statement")

    probe = build_probe(
        statement=statement, imports=imports, preamble=preamble, num_inst=num_inst,
    )

    try:
        prog = Program(probe, Language.LEAN, name="counterexample_check")
        ver = backend.verify(prog, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        logger.debug("counterexample_check verify error: %s", e)
        return CheckResult(passed=True, verdict="inconclusive", reason=f"verify_error: {e}")

    output = (ver.stdout or "") + "\n" + (ver.stderr or "")

    if _looks_refuted(output):
        return CheckResult(
            passed=False,
            verdict="refuted",
            reason="slim_check found a counterexample",
            witness=_extract_witness(output),
            details={"output_excerpt": output[:1500]},
        )

    if ver.outcome == VerificationOutcome.SUCCESS:
        return CheckResult(
            passed=True, verdict="passed",
            reason=f"slim_check ran {num_inst} samples without refuting",
        )

    return CheckResult(
        passed=True, verdict="inconclusive",
        reason="slim_check probe did not compile",
        details={"output_excerpt": output[:800]},
    )
