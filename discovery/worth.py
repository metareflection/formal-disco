"""Multiplicative worth function for heuristics.

Replaces v2's ``interestingness *= boost_or_decay`` with a derived score:

    worth = admit_rate × prove_rate × max(novelty_avg, ε) × (a + b · diff_avg)

where each factor lives in [0, 1] (after Laplace smoothing for the rates).
The construction is multiplicative because each factor is *necessary*:

  - admit_rate: how often the heuristic's proposals survive the cheap checks
    (typecheck, novelty, counterexample). A heuristic that proposes garbage
    has a low admit rate and gets demoted.

  - prove_rate: how often admitted proposals lead to a verified theorem.
    Catches heuristics that admit plausibly-typed conjectures that turn out
    to be unprovable.

  - novelty_avg: mean cosine-distance-to-corpus of admitted proposals. The
    treadmill-killer: a heuristic that keeps proposing Mathlib restatements
    has high admit_rate × prove_rate but near-zero novelty, and so collapses.

  - difficulty_avg: rough proxy for how hard the proofs were (proof line
    count, normalized). Soft-weighted ``a + b · diff_avg`` rather than a raw
    multiplier so heuristics aren't punished for proving easy true things.

The constants are deliberately conservative for Phase 1 — cold-start heuristics
with no admits yet keep a non-zero floor so the scheduler still tries them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorthComponents:
    attempts: int = 0
    admits: int = 0
    proves: int = 0
    novelty_sum: float = 0.0
    difficulty_sum: float = 0.0

    @classmethod
    def from_properties(cls, props: dict) -> "WorthComponents":
        return cls(
            attempts=int(props.get("attempts", 0) or 0),
            admits=int(props.get("admits", 0) or 0),
            proves=int(props.get("proves", props.get("successes", 0)) or 0),
            novelty_sum=float(props.get("novelty_sum", 0.0) or 0.0),
            difficulty_sum=float(props.get("difficulty_sum", 0.0) or 0.0),
        )


# Cold-start floor on novelty so heuristics with no admits aren't immediately killed.
NOVELTY_FLOOR = 0.1
DIFFICULTY_BIAS = 0.3
DIFFICULTY_SCALE = 0.7
SCALE = 4.0  # rough calibration so a "decent" heuristic lands near 1.0


def admit_rate(c: WorthComponents) -> float:
    return (c.admits + 1) / (c.attempts + 2)


def prove_rate(c: WorthComponents) -> float:
    return (c.proves + 1) / (c.admits + 2)


def novelty_avg(c: WorthComponents) -> float:
    if c.admits <= 0:
        return 0.5  # neutral prior
    return max(0.0, min(1.0, c.novelty_sum / c.admits))


def difficulty_avg(c: WorthComponents) -> float:
    if c.proves <= 0:
        return 0.5  # neutral prior
    return max(0.0, min(1.0, c.difficulty_sum / c.proves))


def compute_worth(c: WorthComponents, *, global_proves_seen: bool = True) -> float:
    """Compute a heuristic's worth from its components.

    During the *cold-start* phase — before *any* heuristic in the system has
    been credited with a successful proof — we replace ``prove_rate`` with the
    same neutral prior (0.5) we use for novelty/difficulty when their inputs
    are absent. This stops the kill rule from firing on the four heuristics
    actively producing admits while ProofWorker is still warming up.

    Once any heuristic globally has ``proves > 0``, the standard formula kicks
    in for everyone — including those that have been racking up admits without
    successes, so prove_rate punishes them at that point as intended.
    """
    a = admit_rate(c)
    p = prove_rate(c) if global_proves_seen else 0.5
    n = max(novelty_avg(c), NOVELTY_FLOOR)
    d = DIFFICULTY_BIAS + DIFFICULTY_SCALE * difficulty_avg(c)
    return SCALE * a * p * n * d


def _global_proves_seen(agenda) -> bool:
    """Return True if any heuristic in the agenda has been credited with a prove.

    Reads the private ``_objects`` dict directly because the public Agenda
    protocol only exposes ``get_object(path)``. Same Phase 1 expedient as in
    ``checks/novelty.py``.
    """
    objects = getattr(agenda, "_objects", None)
    if not objects:
        return False
    for obj in objects.values():
        if obj.type != "heuristic":
            continue
        proves = obj.properties.get("proves", obj.properties.get("successes", 0)) or 0
        if int(proves) > 0:
            return True
    return False


async def update_heuristic_worth(
    agenda,
    heuristic_name: str,
    *,
    attempts_delta: int = 0,
    admits_delta: int = 0,
    proves_delta: int = 0,
    novelty_delta: float = 0.0,
    difficulty_delta: float = 0.0,
) -> tuple[float, float] | None:
    """Atomically update a heuristic's worth components and recompute interestingness.

    Returns (before, after) interestingness, or None if the heuristic doesn't exist.
    """
    h_obj = await agenda.get_object(f"heuristic/{heuristic_name}")
    if h_obj is None:
        return None

    components = WorthComponents.from_properties(h_obj.properties)
    components.attempts += attempts_delta
    components.admits += admits_delta
    components.proves += proves_delta
    components.novelty_sum += novelty_delta
    components.difficulty_sum += difficulty_delta

    before = float(h_obj.interestingness)
    new_worth = compute_worth(components, global_proves_seen=_global_proves_seen(agenda))
    factor = (new_worth / before) if before > 1e-6 else new_worth

    await agenda.update_object(
        h_obj.path,
        new_properties={
            "attempts": components.attempts,
            "admits": components.admits,
            "proves": components.proves,
            "novelty_sum": round(components.novelty_sum, 6),
            "difficulty_sum": round(components.difficulty_sum, 6),
            # Mirror proves into the legacy "successes" field so the existing UI
            # and analysis scripts continue to render heuristic success counts.
            "successes": components.proves,
        },
        interest_factor=factor,
    )
    return (before, new_worth)


def difficulty_from_proof(proof_text: str, *, repaired: bool = False) -> float:
    """Rough proof-difficulty proxy in [0, 1].

    Counts non-empty proof lines, capped at 30. Repaired proofs (the LLM had
    to be told its first attempt failed) get a fixed 1.0 — they were hard
    enough to require feedback. Tune in Phase 2 with a real prover-rejection
    signal (e.g., ``aesop`` failure).
    """
    if repaired:
        return 1.0
    if not proof_text:
        return 0.0
    body = proof_text.split(":=", 1)[-1] if ":=" in proof_text else proof_text
    lines = [l for l in body.splitlines() if l.strip() and not l.strip().startswith("--")]
    return max(0.0, min(1.0, len(lines) / 30.0))
