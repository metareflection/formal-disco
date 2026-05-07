# Alignment Probe + Worth-Tweak Experiment — Report

Companion to `REPORT_MATROID.md`. Documents two follow-up experiments on
the matroid discovery run that sharpen the report's central question:
*are the LLM-invented concepts grounded?* and *if not, is the bottleneck
attention or depth?*

Run dates: 2026-05-06.

## TL;DR

The matroid system invented 144 predicates. Lean-driven alignment shows
**only 4 of 110 testable ones (~4%) are α-equivalent to a seed concept**;
the rest are either symbolically novel (63), at a new argument shape (43),
or unparseable (34). Cross-referenced against proved theorems at the end
of the original 200-attempt run: **1 of 29 proved theorems uses
genuinely-novel invented vocabulary**.

A diagnostic on the prove-task pile revealed the dominant limiter: of
109 invented-vocab prove tasks created, the prover attempted **only 1**
(vs 33 of 321 seed-only tasks). Priority was overwhelmingly biased
toward seed-only conjectures.

A worth-tweak experiment (×3.0 boost on prove-task priority for any
conjecture using invented vocabulary, applied retroactively to the
existing pile via a one-off pickle rewrite) was launched. It hit cap at
400 attempts. **Final result: proved-with-invented-vocab grew from 1 →
14, of which 12 are substantive multi-step constructive proofs** — real
matroid theorems including `spanning_circuit_iff_base_insert` (17-line
proof) and `is_loop_iff_k_dependent_one` (13-line proof). Priority was
the bottleneck. With it removed, the proof + proof-repair pipeline
handles depth.

## Setup: alignment toolchain

`align/` and `align_run.py` implement a Lean-driven α-equivalence probe:
for each invented concept produced by a discovery run, generate
candidate iff theorems against canonical concepts (seed or curated
Mathlib wrappers) and ship them to `lake env lean` to prove. Concepts
that prove iff via a tactic ladder (`unfold; rfl`, `unfold; tauto`,
`unfold; aesop`) are tagged as aliases.

Three modes:

* **`vs-seed`** (default): each invented concept tested against the seed pool.
* **`invented-vs-invented`** (`--mode invented-vs-invented`): pairwise
  within the invented pool, transitively closed into synonym clusters.
  Catches multiple LLM-named variants of the same predicate.
* **`vs-canonical`** (`--canonical <path>`): test against a curated
  external pool. `data/canonical_matroid_mathlib.json` provides Mathlib
  predicate wrappers for matroid Phase 2 alignment.

`ground_run.py` cross-references alignment results against proved
theorems to produce the headline metric: how many proved theorems use
*genuinely-novel* invented vocabulary (i.e., concepts that didn't align
to anything canonical).

## Results: Phase 1 (invented vs seed)

`agenda-discovery-matroid.pkl` (200-attempt matroid run) →
`outputs/align-matroid/`.

| | count | %    |
|---|---|---|
| total invented (def/op kind) | 144 | 100% |
| **alias** (iff to a seed concept) | 4 | 2.8% |
| **novel** (no iff with any seed; same arg shape) | 63 | 43.8% |
| **shape_mismatch** (no seed has compatible signature) | 43 | 29.9% |
| **unparseable** (description text or misclassified theorem) | 34 | 23.6% |

The 4 aliases (all of which collapse to seed concepts):

| Invented | Seed | Tactic |
|---|---|---|
| `is_dependent_op` | `is_dep_def` | `unfold; rfl` |
| `is_loop_generalized` | `is_loop_def` | `unfold; rfl` |
| `loop_element` | `is_loop_def` | `unfold; rfl` |
| `loop_as_dual_coloop` | `is_loop_def` | `unfold; aesop` |

`loop_as_dual_coloop` is the most interesting alignment: the system
defined "loop as dual coloop" (a non-trivial equivalence in matroid
theory) and `aesop` proved it iff with the canonical loop predicate.

The 63 "novel" concepts include `dual_indep_op`, `is_cocircuit_op`,
`is_cobase_op`, `coindep_op`, `is_loop`, `is_coloop`, `is_hyperplane_def`,
`is_modular_flat_def`, `is_spanning_circuit`, etc. **Many of these are
Mathlib-aligned**, not truly novel — the seed pool just doesn't include
their canonical counterparts. Phase 2 alignment (vs `canonical_matroid_mathlib.json`)
would reclassify a sizeable chunk.

## Results: grounding (proved theorems × alignment)

`outputs/align-matroid/grounding.md`:

| Category | Count |
|---|---|
| `seed_only` (no invented vocab in statement) | 28 |
| `fully_aliased` (invented vocab, all aliases to seed) | 0 |
| `partially_novel` (≥1 invented concept didn't align) | 1 |

The single `partially_novel` proved theorem is
**`uniform_matroid_circuits_have_uniform_size`**, which uses
`is_uniform_matroid` (a `Matroid α → Prop` predicate at a shape no seed
has).

Headline: **1 of 29 proved theorems contains genuinely-novel invented content**.

## The priority finding

Inspecting the prove-task pile in the 200-attempt pickle (`prove`-typed
tasks classified by whether their conjecture's statement uses any
invented predicate):

| | invented-vocab | seed-only | inv:seed ratio |
|---|---|---|---|
| prove tasks CREATED | 109 | 321 | 1:3 |
| prove tasks ATTEMPTED | **1** | **33** | **1:33** |
| prove DONE | 1 | 22 | |
| prove FAILED | 0 | 10 | |

**The prover only attempted 1 of 109 invented-vocab tasks. The 1 it tried
succeeded.** The system was *generating* invented-vocab conjectures at
roughly 1:3 the rate of seed-only ones, but the prover was *touching*
them at 1:33 — about 11× under-attempted relative to creation rate.

Priority is computed in `_create_concept` (`worker/discovery_worker.py`,
line 274) as `(successes + 1) / (attempts + 2)` — a Laplace-smoothed
heuristic-success-rate. Heuristics that produce mostly seed-only
conjectures get high success rates and so their conjectures dominate the
queue. Born/reflection heuristics that produce invented-vocab
conjectures have lower observed success rates (because the prover never
gets to try their output) and so stay at the bottom.

## Worth-tweak experiment (completed)

A configurable multiplier `invented_vocab_boost` was added to
`DiscoveryWorker`. When the conjecture's statement contains any invented
predicate name, the prove task's priority is multiplied by the boost.
Default 1.0 (off); matroid config sets it to 3.0.

To apply the boost retroactively to the 108 existing invented-vocab
prove tasks already in the queue, `rebalance_invented_vocab.py` was run
once on the pickle: it walks all unsettled prove tasks, identifies those
whose conjecture references an invented predicate, and multiplies their
TaskStatus priority by 3.0. Output: `agenda-discovery-matroid-boosted.pkl`.

The discovery scheduler was resumed against the boosted pickle with the
cap raised to 400. Run hit the cap.

### Final results (400/400 attempts, 200 in resume)

Resume-tail deltas vs pre-resume baseline:

| | t=200 (pre-resume) | t=400 (final) | Δ in resume |
|---|---|---|---|
| prove ATTEMPTED (invented) | 1 | 16 | **+15** |
| prove ATTEMPTED (seed_only) | 33 | 46 | +13 |
| prove DONE (invented) | 1 | 11 | **+10** |
| prove DONE (seed_only) | 22 | 23 | +1 |
| prove FAILED (invented) | 0 | 4 | +4 |
| prove FAILED (seed_only) | 10 | 20 | +10 |
| proved-with-invented theorems | 1 | **14** | **+13** |

The boost achieved parity in attempt counts — invented and seed-only
attempted at roughly the same rate post-resume. The DONE rate flipped:
**10 new invented DONEs vs 1 new seed_only DONE** (most easy seed-only
conjectures had been picked off in the original run, leaving only hard
ones; the unblocked invented-vocab pile was where the easy wins now
lived).

In-resume invented success rate: **10/15 = 67%** (vs seed_only's 1/13 ≈
8% in the resume tail).

### Substantive vs trivial proofs

Of the 14 proved-with-invented theorems:

| Category | Count | Examples |
|---|---|---|
| **Substantive** (multi-line constructive proof) | **12** | `spanning_circuit_iff_base_insert` (17 lines), `uniform_matroid_circuits_have_uniform_size` (19), `is_loop_iff_k_dependent_one` (13), `spanning_circuit_rank_identity` (15) |
| **Trivial** (`Iff.rfl` / `rfl` definitional restatement) | 2 | `self_dual_iff_double_dual_fixes`, `closure_coclosure_dual_conjugation` |

The 12 substantive proofs are real matroid arguments. For example,
`spanning_circuit_iff_base_insert` chains 6+ Mathlib lemmas
(`hC.diff_singleton_indep`, `Matroid.IsCircuit.closure_diff_singleton_eq`,
`Matroid.spanning_iff_closure_eq`, `hindep.isBase_of_spanning`, etc.)
into a base-deletion characterization of spanning circuits. This is the
shape of proof a textbook would give.

### Diagnosis

Priority was the bottleneck. The boost unblocked the invented-vocab
pile and the existing **proof_repair worker** handled the depth
challenge: 26 proof_repair tasks completed in this run, taking initial
prove failures (e.g., the `spanning_circuit_*` cluster I had previously
flagged as failures at t=224) and constructing successful multi-step
proofs on retry.

Earlier framing of "depth as residual bottleneck" came from observing
3 failures at t=224 before repair had time to work them. By cap, those
same conjectures were proved.

## Implications for "humanely meaningful" output

REPORT_MATROID.md ended with: *"the discovery system produces internally
consistent output that doesn't ground in shared vocabulary."* The
alignment + worth-tweak experiments add three sharpenings:

1. **Most invented vocabulary is not seed-aliased.** 4 of 110 testable
   (~4%) collapse back to seeds. The bulk is symbolically distinct
   (though many will likely Mathlib-align — a Phase 2 hypothesis to test).

2. **At the original 200-attempt cap, 28/29 proved theorems lived in
   seed vocabulary.** With the worth-tweak resume, this shifts to **43
   of 57** — still the majority, but no longer overwhelming. The system
   can produce humanely-meaningful invented-vocab content; it just
   needs the prove step to be told to look at it.

3. **The proposer-verifier gap is fixable by re-prioritization.** What
   looked like a structural depth limit was a priority-attention
   limit. Once the prover gets the floor, the existing
   prove + proof_repair pipeline produces 12 substantive multi-step
   matroid proofs out of 14 invented-vocab DONEs (86% substantive).
   "Reasonable reflection requires the verifier to keep pace" is the
   right framing, and the verifier *can* keep pace given attention.

### What the substantive proofs actually demonstrate

The system has not constructed matroid theory from first principles. The
12 substantive proofs heavily invoke Mathlib's existing matroid library
(`Matroid.IsCircuit.closure_diff_singleton_eq`,
`Matroid.spanning_iff_closure_eq`, `hC.diff_singleton_indep`, etc.).
What it has done is *navigate* that library — choose which lemmas to
chain, in what order, with what intermediate predicates — to discharge
non-trivial conjectures stated in invented vocabulary. That's real proof
engineering work, but it's "library composition," not "theory building
from foundations."

This nuance matters for the keynote framing: a mathematician inspecting
`spanning_circuit_iff_base_insert` would recognize it as substantive
matroid reasoning, but they'd also notice the proof leans on
Mathlib-canonical lemmas. The corpus is "Mathlib-style matroid theorems
rephrased in LLM-named vocabulary, with proofs that compose Mathlib
primitives." That's not nothing — but it's not the same as the system
*deriving* base-cobase duality.

## What's left to try

- **Phase 2 alignment** (vs `canonical_matroid_mathlib.json`). Likely
  reclassifies a chunk of the 63 "novel" concepts as Mathlib-aliases,
  giving a sharper count of what's actually new content.
- **Invented-vs-invented alignment** for synonym clustering. Catches the
  multiple-loopless-variants case where no seed has the relevant shape.
- **Re-run grounding on the boosted pickle.** The grounding tool was
  run against the original 200-attempt pickle. With 14 proved-with-invented
  in the 400-attempt boosted pickle, re-running grounding would update
  the headline metric.
- **Bigger run with the boost on.** 14 proved-with-invented over a
  200-attempt resume is a small sample. cap=1000+ with the boost on
  from t=0 (rather than retroactively) would let us track invented-vocab
  yield as the corpus matures.
- **Disentangle Mathlib-composition from theory-building.** The 12
  substantive proofs lean on Mathlib lemmas. A version of grounding
  that scores proofs by "fraction of proof body that's Mathlib calls
  vs. inline reasoning" would tell us whether the system is composing
  or constructing.

## Files

| | |
|---|---|
| Alignment toolchain | `align/__init__.py`, `align/probe.py`, `align/cluster.py`, `align/grounding.py` |
| Entry points | `align_run.py`, `ground_run.py` |
| Mathlib canonical pool | `data/canonical_matroid_mathlib.json` |
| Worth-tweak code | `worker/discovery_worker.py` (`invented_vocab_boost` param + `_invented_concept_names` helper) |
| Rebalance utility | `rebalance_invented_vocab.py` |
| Output | `outputs/align-matroid/{alignment.json,alignment.md,grounding.md}` |
| Boosted pickle | `agenda-discovery-matroid-boosted.pkl` |

## One-line takeaway

Lean-checking the LLM-invented matroid vocabulary against seed vocabulary
shows that ~96% of it is symbolically novel; the prover overwhelmingly
ignored that novel vocabulary at the original priorities; with a ×3
priority boost on invented-vocab prove tasks, the prover and its
proof_repair partner produced 14 verified theorems in invented matroid
vocabulary, of which 12 are substantive multi-step constructive proofs
that compose Mathlib's matroid library — confirming that priority, not
proof depth, was the dominant limiter on humanely-meaningful output.
