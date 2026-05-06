# Matroid Discovery Run — Report

A 200-attempt run of the Eurisko-style Lean discovery system on a new
domain (matroid theory), designed to test whether moving to a less
LLM-trodden domain shifts the system's output from "first-chapter
textbook recitation" toward something a mathematician would call
meaningful.

Run date: 2026-05-05.

## TL;DR

Domain obscurity helps at the *proposal* stage (the system invented
144 matroid concepts and proposed 225 conjectures using its own
invented vocabulary, including correct statements of base-cobase
duality and Whitney-style hyperplane/cocircuit complementarity) but
not at the *proving* stage (only 1 of 29 proved theorems uses an
invented concept). The deeper lesson is that even *correct + interesting*
LLM-generated concepts are **ungrounded**: the system has no automated
way to know whether `is_simple_matroid` matches Oxley's definition,
Mathlib's `Matroid.Simple`, or neither.

## Hypothesis

REDESIGN.md identified the dominant failure mode of prior runs (nat,
group_theory, modarith) as "the system isn't discovering — it's
reciting." First-chapter Peano arithmetic is exactly what the LLM has
seen thousands of times, so its proposals are textbook-trivial.

Hypothesis: pick a domain where the LLM has *less* training data — not
so little that it hallucinates everything, but enough that pattern-matching
to a known one-line proof isn't the path of least resistance — and the
system's output will shift from textbook recitation toward genuine
discovery.

Matroid theory is a good fit: textbook results exist (so the LLM has
*some* prior), Mathlib has the type but not the bulk of standard theory
(`Mathlib/Combinatorics/Matroid/`), counterexamples are concrete on
finite examples, and the canonical "deep" result — Whitney's
circuit-cocircuit duality — is exactly the kind of thing a mathematician
would call non-obvious.

## Setup

### Seed

`data/seed_matroid_theory.json`: 11 concepts (8 operations/definitions
+ 3 seed theorems). Designed to give the system enough vocabulary to
reason but not enough to short-circuit interesting discovery:

| Seeded | Deliberately omitted |
|---|---|
| `indep_op`, `is_base_op`, `is_circuit_op`, `closure_op`, `dual_op` | `dual_dual_thm` (`(M✶)✶ = M`) |
| `is_loop_def`, `is_flat_def`, `is_dep_def` | Cocircuits, cobases, coloops as named concepts |
| `indep_empty_thm`, `closure_idempotent_thm`, `isBase_subset_ground_thm` | Rank function (let it surface if useful) |

The dual operation was seeded but its involutive nature was not, leaving
room for the system to discover it.

### Config

`config/scheduler/discovery_matroid_aws.yaml`: clone of the modarith
discovery config, with `domain: matroid_theory`, Opus 4.7 throughout
(implementer, prover, fixer, reflector), and `enable_counterexample_check:
false` (matroids are abstract; no SampleableExt instances available, so
slim_check would always be inconclusive — same call as group_theory).

Workers: ConceptSeeder → DiscoveryWorker → ProofWorker → LLMFixer →
ProofRepairWorker → ReflectionWorker, in round-robin with `turn_fuel: 5`.

### Why the worth tweak from the prior conversation was *not* applied

The plan included a cross-cluster worth modulation. Skipped on first
run to test the obscurity hypothesis with the unmodified pipeline. The
matroid run is the baseline; any worth tweak would be a follow-on.

## Run

`agenda-discovery-matroid.pkl`. 200 attempts, ~70 minutes wall clock.
Background command exited cleanly when the cap was hit.

### Top-line numbers

| Metric | Value |
|---|---|
| total_attempts | 200/200 |
| clock | 12,034 |
| invented concepts (definition/operation) | **144** |
| seed concepts | 8 |
| theorems (`kind=theorem`) | 41 |
| **proved theorems** (no `sorry`) | **29** |
| **proved with invented vocab** | **1** |
| proved with multistep proof body | 28 |
| proved with 1-line discharge (`by simp`) | 1 |
| conjectures (`kind=conjecture`, unproven) | 679 |
| **conjectures using invented vocab** | **225** (33%) |
| born heuristics (reflection-admitted) | 27 |
| seed heuristics | 9 |
| prove tasks: DONE / FAILED | 23 / 10 |
| proof_repair tasks: DONE / FAILED | 16 / 8 |

### Sample proved theorems

The system proved 29 theorems including:

- **`uniform_matroid_circuits_have_uniform_size`** — the only proved
  theorem in the run that uses an invented concept (`is_uniform_matroid`,
  defined ad-hoc by the system as
  `∀ S ⊆ M.E, M.Indep S ↔ S.encard ≤ M.eRank`). Multi-step proof with
  a sublemma extracting circuit cardinality.
- **`base_iff_dual_base_compl`** — `M.IsBase B ↔ M✶.IsBase (M.E \ B)`.
  Real Whitney-style duality statement; proof uses Mathlib's
  `compl_isBase_dual` for the forward direction, then unfolds via
  `Matroid.dual_dual` for the backward.
- **`dual_dual_involution`** — `(M✶)✶ = M`. Proof: `by simp`. Closes
  because `Matroid.dual_dual` is a Mathlib simp lemma.
- **`loop_not_in_any_base`** — 4-line constructive proof.
- **`no_empty_circuit`** — 2-line proof via contradiction.

### Sample unproven invented-concept conjectures

The system proposed 225 conjectures whose statements use vocabulary the
system itself invented. These remain unproven. Examples:

- **`hyperplane_complement_is_cocircuit`** — *the Whitney duality
  statement.* If proved, this would be the canonical non-trivial
  matroid result.
- **`cobase_iff_compl_base`** — cobases are complements of bases.
- **`cocircuit_iff_complement_cobase_hyperplane`** — three-way
  characterization tying cocircuits, cobases, and hyperplanes.
- **`coloop_iff_in_every_base`** — coloop characterization.
- **`simple_implies_loopless`** — simple matroid ⇒ loopless.
- **`circuit_cocircuit_intersection`**, **`loop_iff_dual_coloop`**,
  **`hamiltonian_circuit_complement_is_base_of_dual`**.

These are non-trivial textbook results stated in invented vocabulary
(`is_cobase_op`, `is_cocircuit_op`, `is_simple_matroid`, etc.). The
system *generated* them; the prover did not reach them.

### Invented vocabulary the system built

A representative slice of the 144 invented concepts:

- **Dual hierarchy**: `is_cocircuit_op`, `is_cobase_op`, `coindep_op`,
  `coloop_as_dual_loop`, `cocircuit_as_dual_circuit`,
  `dual_indep_op`. The system rebuilt the dual concepts on top of the
  seeded `dual_op`.
- **Special matroid classes**: `is_uniform_matroid`,
  `is_loopless_matroid`, `is_simple_matroid`, `self_dual_matroid`,
  `is_spanning_base_specialized`.
- **Equivalent formulations**: `is_loop` (defined as `M.IsCircuit {e}`,
  which is α-equivalent to the seeded `is_loop_def` but the system
  doesn't know that), `is_loop_specialized`, `is_coloop_specialized`,
  `loopless_matroid_on_ground` (a third loopless variant).

## Analysis

### What worked

**Domain obscurity shifted the proposal stage decisively.** The
matroid run produced 225 invented-concept conjectures (33% of the
total conjecture pile). Compare to `agenda-phase2-modarith.pkl` at
similar attempt counts (176 attempts, 33 proved): the modarith
output is dominated by Peano-trivial restatements with no concept
invention to speak of. The qualitative shift is real and large.

**Reflection produced 27 new heuristics.** The reflection step
admitted 27 born heuristics through the soundness gate, none of
which appear in the seeded list. So the heuristic pool grew with
domain-specific intuitions.

**Concept invention is correct, mostly.** Spot-checking the 144
invented concepts: most are α-equivalent to standard matroid
definitions (`is_loopless_matroid`, `is_simple_matroid`, `is_uniform_matroid`
all match textbook definitions). A few are duplicates of seeded
concepts under different names. None spotted so far that are wildly
wrong.

### What didn't

**The proof step did not follow.** Of 29 proved theorems, only 1
uses invented vocabulary. The other 28 are stated in seed vocabulary
and proven by chains of Mathlib lemmas. The 225 invented-concept
conjectures sit in the pile.

**The proof bias is structural.** When a conjecture lives entirely
in seeded vocabulary, the LLM-as-prover finds a Mathlib lemma and
discharges. When the conjecture uses invented predicates, the proof
must (a) unfold the invented definition, (b) chain multiple Mathlib
lemmas. The latter regime is harder and the prover stalls. The path
of least resistance is "stay in seed vocabulary; quote Mathlib."

**Most proofs are *real* but *shallow*.** 28 of 29 proved theorems
have multi-line bodies, not just `by simp`. So the proofs are doing
work — but the work is library navigation rather than constructive
reasoning. A typical "interesting" proved theorem
(`base_iff_dual_base_compl`) chains 2–3 Mathlib lemmas under existing
infrastructure. None of the 29 proved theorems builds new
intermediate machinery.

## The deep lesson

Even where the system did succeed (proposing meaningful matroid
conjectures in invented vocabulary), the output is **ungrounded**.
Each invented concept is a fresh definition the LLM authored. The
system has no automated way to know whether a given invented
predicate is:

1. The **standard** definition under a non-canonical name. Example:
   `loopless_matroid_on_ground` defined as
   `∀ e ∈ M.E, M.Indep {e}` is α-equivalent to
   `∀ e ∈ M.E, ¬ is_loop_def M e` (the canonical loopless
   definition), but the system has invented both as separate concepts
   and treats them as unrelated.
2. A **subtly off** definition that coincides with the standard one
   on most cases but diverges on edge cases (e.g. quantification
   restrictions over the ground set vs over all of α; behavior on
   infinite ground sets).
3. A **plausible-looking but wrong** definition the LLM
   hallucinated. Lean catches ill-typed definitions but not
   ill-conceived ones.
4. A **genuinely new** concept that's coherent but isn't matroid-canonical.

Whether the corpus is meaningful depends on which of these dominates.
If most invented concepts are (1) or (2), we have a private dialect
that overlaps with the discipline's vocabulary but isn't intelligible
to it; if (3) is non-trivial, theorems about those concepts are
correct in Lean but incorrect-as-mathematics; if (4) is genuine,
we have something potentially valuable but currently unreviewable.

What's mitigating in this run:
- The seeded vocabulary is canonically aligned (`indep_op := M.Indep`,
  etc.). Most invented concepts are compositions of seeded ones, so
  their content can in principle be unfolded back to canonical
  predicates.
- Lean type-checks invented concepts symbolically; outright nonsense
  doesn't get past the parser.

What isn't mitigating:
- No automated alignment check: nothing tests "does `is_simple_matroid`
  match `Mathlib.Combinatorics.Matroid.Simple`?" That's a
  definitional-equivalence question, hard in general but tractable
  for many concrete cases (rewrite the invented def into Mathlib
  primitives and use `decide`/`aesop` to test the iff).
- No automated review: a mathematician spotting cases (3) and (4)
  has the corpus; the system doesn't.
- Even when the invented concept *is* α-equivalent to a Mathlib one,
  theorems stated in the invented vocabulary live in a parallel
  universe — locally coherent, globally illegible.

## Implications

The actionable framing is: **a discovery system whose output you want
a mathematician to read needs a definitional-grounding step woven into
the loop, not bolted on at the end.**

Specifically, when an invented concept is created, the system should:

1. Compute syntactic-or-semantic distance to existing canonical
   definitions (Mathlib + the seed corpus).
2. If close enough: alias the invented concept to the canonical one
   and drop the duplicate.
3. If genuinely new: flag it explicitly as "novel definition" so
   downstream theorems about it carry that flag, and reviewers know
   the corpus is making a definitional claim, not just a theorem.

Step (1) is genuinely hard. Embedding-based similarity is necessary
but not sufficient — the failure modes of this experiment (multiple
invented variants of "loopless matroid" that don't realize they're
synonyms) won't be caught by syntax. You need an *equivalence-check*
step that's at least decidable for predicates over standard structures
— probably some combination of `aesop`/`decide` over a normal-form
representation.

In the absence of (1), the discovery system produces internally
consistent output that doesn't ground in shared vocabulary. The
result is a parallel mathematics: each run produces a private dialect.

## Falsifiability for next runs

Two follow-on experiments would tighten this picture:

**Experiment 1 — Worth tweak.** Bias proof-task priority toward
conjectures that use invented vocabulary. Concretely: in
`discovery/worth.py`, add a per-task multiplier when the conjecture
statement contains any non-seed concept name. Re-run the matroid
checkpoint with cap=400 and observe whether more of the 225 invented-
concept conjectures convert to proved. Tests whether the prove-stage
bottleneck is priority (fixable) or depth (structural).

**Experiment 2 — Definitional alignment probe.** For each of the 144
invented concepts, write a pairwise check against the seeded definitions
(α-equivalence modulo unfolding). Output: a clustering where
`is_loopless_matroid`, `loopless_matroid_on_ground`, and any other
synonyms collapse into one bucket. Quantifies how much of the
"discovery" is actually rediscovery of seed concepts under different
names. If the collapse rate is high, we have less new content than
the 144-concept count suggests.

## Files

| | |
|---|---|
| Seed | `data/seed_matroid_theory.json` |
| Scheduler config | `config/scheduler/discovery_matroid_aws.yaml` |
| Pickle | `agenda-discovery-matroid.pkl` |
| Discovery module | `discovery/` (unmodified) |
| Mathlib path | `LeanDisco/.lake/packages/mathlib/Mathlib/Combinatorics/Matroid/` |

## One-line takeaway

The system *can* propose meaningful theorems in a less-trodden domain;
the bottleneck has moved from imagination to grounding, and grounding
LLM-invented vocabulary back to canonical definitions is itself an
unsolved problem.
