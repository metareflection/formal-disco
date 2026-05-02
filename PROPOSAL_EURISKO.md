# Proposal: Verified Eurisko in `formal-disco`

A redesign that re-ports the v2 eurisko layer back into `formal-disco/` and reshapes
it as a proposer/checker system, so it stops converging on textbook material.

## 0. Where we are

`formal-disco/` currently has the agenda + worker scaffolding (LLMImplementer, LLMFixer,
EditorWorker, ReadmeInspiredIdeaGenerator) but **none of the eurisko code**:
`discovery/` is empty, and `worker/` is missing `concept_seeder.py`,
`discovery_worker.py`, `proof_worker.py`, `proof_repair_worker.py`,
`reflection_worker.py`. The `agenda-discovery-*.pkl` files are leftover checkpoints
produced by `formal-disco-eurisko2/`, where the code still lives.

So this proposal has two parts:

- **Part A — Re-port** (one afternoon): bring v2's eurisko layer back into
  `formal-disco/`, clean up the Vertex/AWS split, verify it runs from one of the
  existing `agenda-discovery-*.pkl` checkpoints.
- **Part B — Redesign** (the substance): replace v2's worth function, slot model, and
  heuristic-as-prompt-template with a proposer/checker architecture so reflection earns
  its keep.

Part A is mechanical and gives a working baseline. Part B is what makes the system
humanely worthwhile.

## 1. Lineage

| project | what it contributed |
|---|---|
| `eurisclo/` | 29 heuristics with rich slot structure (`if-potentially-relevant`, `if-truly-relevant`, `then-compute`, `then-define-new-concepts`, `then-add-to-agenda`); worth as discrete int; heuristics as Lisp lambdas — true reflective rewrites. |
| `formal-disco-eurisko1/` | First port to formal-methods discovery (Dafny + Lean + Verus). Distillation, fixer, eval, lemma-repair pipelines. Eurisko ideas mixed in but not factored. |
| `formal-disco-eurisko2/` | Clean factoring: `discovery/{__init__,heuristics_seed,prompts}.py` + 5 eurisko workers. 9 heuristics ported as prompt templates, worth as Laplace-smoothed success rate, kill rule at attempts≥10 and rate<5%. Writes proved theorems into `../LeanDisco`. |
| `formal-disco/` | Today: agenda + base workers, no eurisko layer. Has `REDESIGN.md` analyzing why v2 was boring. |

What v2 dropped from eurisclo:
- The slot split (`if-potentially-relevant` / `if-truly-relevant` / `then-compute` / etc.)
  collapsed into a single prompt `template` field. This makes heuristic-birth
  meaningless: a "new heuristic" is just a new prompt string with no structural
  difference from the old ones.
- Heuristic actions as **code**. v2 heuristics propose mutations only by asking the
  LLM nicely. eurisclo heuristics could literally rewrite the system — that's the
  reflective move "license to hack" justifies, and it's what we want back.

## 2. Why v2 was boring (diagnosis)

From `EURISKO_STATUS.md`: 309 concepts generated from 15 nat seeds, 7/237 conjectures
proved, all 7 via embedded inline proofs from the discovery LLM. `direct_tactic_proof`
0/14, `structured_proof` 0/13. Output is Peano arithmetic, all already in Mathlib.

The structural reason: **the only checker is "Lean accepts the proof."** That checker
has no opinion about novelty, mathematical interest, or self-modification soundness.
Worth = `successes / attempts` therefore rewards a heuristic that produces 50 trivial
`omega`-provable facts over one that produces 2 hard conjectures that fail. The system
optimizes throughput of easy results, which is what "moving slowly though surely but
not very interestingly" means.

Pipeline issues stack on top (round-robin doesn't distinguish cheap from expensive
work; ProofWorker doesn't feed Lean errors back to LLMFixer; LLM hallucinates Mathlib
imports) — `REDESIGN.md` covers these. They're fixable, but fixing them just makes the
treadmill faster. The treadmill itself is the problem.

## 3. Thesis: the proposer/checker asymmetry

Reflective systems were unreasonable because the meta-level could do anything and
checking was out of reach. That has changed. Verifiers now check rich fragments, and
LLMs can propose modifications *together with the proofs the checker requires*. The
asymmetry — proposing is creative and untrusted, checking is mechanical and trusted —
is what lets reflection be safe.

Applied to Eurisko: every heuristic is a **proposer/checker pair**.

- The **proposer** is an LLM call that produces a candidate mutation (new concept,
  new conjecture, new proof, new heuristic, new prompt, new search policy).
- The **checker** is a small machine-grade test that decides whether the candidate is
  admitted. The checker is what v2 lacks.

Checker is *not* "ask another LLM." That collapses the asymmetry. Checkers must be
mechanical: typecheck, decide, counterexample search, signature dedup, entropy delta on
a held-out set, etc.

## 4. Proposed architecture

### 4.1 Heuristic = proposer + checker

```python
@dataclass
class Heuristic:
    name: str
    kind: Literal["concept", "conjecture", "proof", "reflection", "meta"]

    # cheap structural filter (eurisclo's if-potentially-relevant)
    applies_to: Callable[[Object], bool]

    # propose a mutation: LLM call + parser
    propose: Callable[[Context], list[Candidate]]

    # mechanical admission test (the asymmetry's load-bearing piece)
    check: Callable[[Candidate, Agenda], CheckResult]

    # commit the admitted candidate to the agenda (eurisclo's then-add-to-agenda)
    commit: Callable[[Candidate, Agenda], None]

    # worth state
    attempts: int
    admits: int           # passed check
    proves: int           # downstream theorem proved that traces back here
    novelty_sum: float    # accumulator over admitted candidates' novelty scores
```

The slot split matters because heuristic-birth is now structurally meaningful: a new
heuristic has different `applies_to`, `propose`, `check`, or `commit` — not just a
different prompt string.

### 4.2 Five concrete checkers

These are the checkers Part B needs day one. Each is small and mechanical.

1. **TypecheckCheck** (concepts, conjectures): Lean compiles the declaration with
   `sorry` as the body. Already in v2's pipeline; keep it.

2. **CounterexampleCheck** (conjectures): try `decide`, `#eval` on a small stratified
   sample of inputs, and a QuickCheck-style random fuzz with type-driven generators.
   If a counterexample is found, the conjecture is *refuted* (still useful — record
   the witness, ask the LLM to refine the statement). If none found, the conjecture
   *passes* this gate and proceeds to proof. Filters out the false-and-easy class
   that wastes Opus calls.

3. **NoveltyCheck** (concepts, conjectures): novelty score = embedding distance from
   the candidate's statement to the union of (a) Mathlib's declaration index, (b)
   `LeanDisco`'s already-proved corpus, (c) the agenda's existing concepts (signature
   dedup is the floor; embedding is the ceiling). Reject below a threshold, or use
   the score in worth. Mathlib-aware novelty is the single biggest leverage point —
   it's what stops the system from "discovering" `Nat.succ_injective` for the
   thousandth time.

4. **HeuristicSoundnessCheck** (new heuristics): admits a proposed new heuristic only
   if (i) its `applies_to` is non-empty on the current agenda, (ii) its `propose` on
   a held-out concept-set produces candidates that pass `TypecheckCheck` at a rate
   ≥ ε, (iii) the entropy of admitted-candidate features doesn't drop below the
   pre-existing heuristic pool's, (iv) it doesn't subsume an existing heuristic by
   prompt-similarity above a threshold. This is the gate that makes
   reflection-of-the-meta-level safe. Implementation is a fixed evaluation harness
   that runs the new heuristic against ~20 held-out concepts; cheap.

5. **PolicyCheck** (search-policy mutations): if a heuristic proposes a change to the
   scheduler (priority weights, fuel allocation, worker mix), the check is a sandboxed
   replay over the last K agenda ticks: does the proposed policy improve a target
   metric (proves/hour, novelty/hour) without violating safety invariants
   (no-starvation, no-deadlock, bounded-queue-growth)? Reject if not. This is where
   the system gets to rewrite its own scheduler — Lenat's bet, with a soundness gate.

### 4.3 Worth function

Replace `successes / attempts` with:

```
worth(h) = (admits / attempts)         # cheap-check pass rate
         × (proves / max(admits, 1))   # downstream usefulness
         × novelty_avg(h)              # mean novelty of admitted candidates
         × difficulty_avg(h)           # mean proof difficulty (tactic count or
                                       # rejection-by-`exact?`/`aesop` proxy)
```

All four factors are bounded in [0, 1] after normalization. Multiplicative because
each is *necessary*: a heuristic that's prolific but boring (low novelty), or novel
but unprovable (low proves), or hard-but-trivially-false (low admits) all collapse
to ~0. This is what stops the treadmill.

Initial values use Laplace smoothing as v2 did. Decay is unchanged (kill at attempts
≥10 and worth below threshold).

### 4.4 Scheduler

Replace round-robin with **expected-value-per-fuel** scheduling: each worker
estimates the cost of its next task (Lean compile is ~15-30s, Opus call is ~30-120s,
Sonnet call is ~5-20s) and the expected worth contribution. Scheduler picks the
worker with the highest `expected_worth_delta / expected_cost`. Falls back to
round-robin if estimates are unavailable.

This is `REDESIGN.md`'s "priority-aware scheduling" — already on the docket; it
becomes load-bearing once worth is multiplicative because cheap checkers should run
much more often than expensive proof attempts.

### 4.5 The reflection loop, redrawn

```
ConceptSeeder ──→ initial concepts + heuristics
                       │
                       ▼
            ┌──→ DiscoveryWorker ──┐
            │      proposer: LLM   │
            │      check: typecheck + counterexample + novelty
            │                      │
            │      admitted ───────┼─→ ProofWorker (with LSP feedback)
            │                      │      check: Lean accepts proof
            │                      │
            │      refuted/proved/failed
            │             │
            │             ▼
            │     ReflectionWorker
            │      proposes: new concepts, new heuristics, scheduler tweaks
            │      checks: HeuristicSoundnessCheck, PolicyCheck
            │             │
            └─────────────┘  (admitted heuristics enter the heuristic pool;
                              admitted policies update the scheduler)
```

The cycle through ReflectionWorker is the reflective tower step — but now every arc
into the meta-level passes through a mechanical check.

## 5. AWS migration

Mostly plumbing. `config/llm/aws*.yaml` and `config/scheduler/readme_ideas_aws*.yaml`
already exist.

- Add `config/scheduler/discovery_lean4_aws.yaml` mirroring v2's
  `discovery_lean4_vertex.yaml`, swapping `llm: vertex-*` → `llm: aws-*`.
- Verify Bedrock model IDs for the Sonnet/Opus tiers v2 assumes
  (`vertex-sonnet-4.5`, `vertex-opus-4.5/4.6` → AWS equivalents in
  `config/llm/aws-sonnet-4.5.yaml`, `aws-opus-4.5.yaml`).
- Set `AWS_REGION` and credentials per Bedrock norms. No code changes expected.

Document the AWS path as the default in the redesigned README; keep Vertex as an
alternate.

## 6. Roadmap

### Phase 0 — Re-port (1 afternoon, no design risk)

1. Copy `formal-disco-eurisko2/discovery/` → `formal-disco/discovery/`.
2. Copy `formal-disco-eurisko2/worker/{concept_seeder,discovery_worker,proof_worker,proof_repair_worker,reflection_worker}.py` → `formal-disco/worker/`.
3. Copy `formal-disco-eurisko2/data/seed_*.json` → `formal-disco/data/`.
4. Copy `formal-disco-eurisko2/config/scheduler/discovery_*.yaml` → `formal-disco/config/scheduler/`.
5. Resolve any drift in `agenda.py` / `language/` between v2 and current
   `formal-disco/` (likely small).
6. Add `discovery_lean4_aws.yaml`. Smoke-test resume from `agenda-discovery-group.pkl`.

Acceptance: same command runs and produces the same kind of agenda growth as v2.

### Phase 1 — Cheap checks first (≈1 week)

1. Implement `CounterexampleCheck` (Lean `decide`, `#eval`, type-driven fuzz).
2. Implement `NoveltyCheck` against Mathlib + LeanDisco (embedding index, FAISS or
   sklearn).
3. Switch to multiplicative worth (`admits × proves × novelty × difficulty`).
4. Wire ProofWorker → LLMFixer with the actual Lean error output (`REDESIGN.md` calls
   this the highest-leverage fix; it's also the one we already have a worker for).

Acceptance: re-run the nat-seed experiment. Concrete success criterion — at least one
of these:
- Median novelty score of admitted conjectures rises ≥ 0.3 vs v2 baseline.
- A non-`omega`-provable theorem gets proved.
- The system refutes at least one false conjecture before proof attempt.

### Phase 2 — Slot redesign and meta-level (≈2 weeks)

1. Refactor `Heuristic` to the proposer/checker dataclass above. Migrate the 9 v2
   heuristics into the new shape (each gets an explicit `applies_to` and `check`;
   `propose` wraps the existing prompt template).
2. Implement `HeuristicSoundnessCheck` with the held-out concept harness.
3. Let `ReflectionWorker` actually propose new heuristics (not just new conjectures).
   Admitted heuristics enter the pool.

Acceptance: at least one heuristic gets *born* and contributes a proved theorem
without being killed within 50 attempts.

### Phase 3 — Self-rewriting scheduler (≈2 weeks, highest risk)

1. Implement `PolicyCheck` (sandboxed agenda replay).
2. Add a `meta` heuristic that proposes scheduler-policy mutations.
3. Run continuously; evaluate whether the system's scheduler diverges from
   round-robin in ways the policy gate admits.

Acceptance: scheduler policy at week 3 is measurably different from initial and
beats it on `proves_per_hour × novelty_avg`.

### Phase 4 — NL meta-level (the ClaimCheck / "guardians" thread)

Targets stated in NL ("find a non-Mathlib lemma about partial orders"); a
NL→spec checker rejects restatements of known results. Out of scope for the
initial design doc; sketch later.

## 7. What we keep from v2 verbatim

- Agenda + worker/object/task model.
- Lean verification as ground truth; LeanDisco as the cumulative library.
- Hydra config layout; AWS/Vertex/vLLM swappability via `config/llm/`.
- Distillation hooks (every prompt/response/outcome triple is still recorded;
  proposer-LLM SFT data is unchanged in shape).
- `REDESIGN.md`'s direction list — directions B (Mathlib-gap-filling), D
  (counterexample), F (LSP-interactive proving) all slot in as concrete checkers
  or workers under the proposer/checker frame.

## 8. What we drop

- `worth = successes / attempts`. Replaced by the multiplicative form.
- The single-`template` heuristic shape. Replaced by the slotted dataclass.
- Round-robin scheduling. Replaced by EV-per-fuel (with policy mutation in Phase 3).
- Open-ended discovery from minimal seeds with no novelty filter. Without
  `NoveltyCheck`, the system *will* re-derive Mathlib's first chapter; with it,
  every admit has to clear the bar.
- Heuristic birth as "new prompt string." Replaced by code-level
  proposer/checker/commit and `HeuristicSoundnessCheck`.

## 9. Open questions

- **Granularity of `HeuristicSoundnessCheck`'s held-out set**: 20 concepts is a
  guess. Too small → noisy admit rates; too big → expensive to run on every
  proposal. Probably wants to be a config knob and a follow-up experiment.
- **Embedding model for `NoveltyCheck`**: Mathlib statements are syntactically
  noisy; do we embed the statement text, the elaborated term, or a normalized
  signature? Start with text + a normalized signature, compare empirically.
- **Difficulty proxy**: tactic count is rough; `aesop`/`exact?` rejection is
  better but slower. Use tactic count as a fallback when an external prover
  rejection is too expensive to run live.
- **NL meta-level sequencing**: ClaimCheck-style NL targets could be Phase 1.5
  if we want a humanely-worthwhile demo before the full slot redesign lands.
  Tradeoff: ClaimCheck without `HeuristicSoundnessCheck` only changes what the
  system reaches for, not what it admits.

## 10. Why this is the lineage

- From eurisclo: slots, code-level then-actions, the willingness to let the
  meta-level rewrite itself.
- From eurisko1: the formal-methods substrate (Lean/Verus/Dafny verifier as the
  ground checker).
- From eurisko2: the clean factoring (discovery/, four workers, LeanDisco library,
  Hydra configs).
- New: every meta-level move is gated by a mechanical check, so reflection earns
  back the trust that 3-Lisp/Brown/Blond/Black couldn't get from static reasoning.

The bet: with the soundness gate in place, "heuristics propose new heuristics" is
not havoc — it's the thing the system needs to stop being a treadmill of
already-known facts.
