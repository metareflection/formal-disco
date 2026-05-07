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
or unparseable (34). Cross-referenced against proved theorems: **1 of 29
proved theorems uses genuinely-novel invented vocabulary**.

A separate diagnostic on the prove-task pile revealed the dominant
limiter: of 109 invented-vocab prove tasks created, the prover attempted
**only 1** (vs 33 of 321 seed-only tasks). Priority was overwhelmingly
biased toward seed-only conjectures.

A worth-tweak experiment (×3.0 boost on prove-task priority for any
conjecture using invented vocabulary) was launched to test whether
priority alone was the bottleneck. Early results (224/400 attempts):
**all 4 new prove attempts since resume went to invented-vocab tasks**,
producing 1 new proved theorem (`self_dual_iff_double_dual_fixes`). But
inspection shows the 1 successful proof is `Iff.rfl` (a trivial
definitional restatement) while 3 attempted invented-vocab proofs failed
on substantive matroid content (`spanning_circuit_iff_base_insert` and
two corollaries). So priority was only *part* of the bottleneck —
**depth-of-proof is the other part**, and it isn't fixed by re-prioritization.

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

## Worth-tweak experiment (in flight)

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
cap raised to 400. Run still in flight at time of writing.

### Snapshot at 224/400 attempts (24 since resume)

| | t=200 (pre-resume) | t=224 (now) | Δ |
|---|---|---|---|
| prove ATTEMPTED (invented) | 1 | **5** | +4 |
| prove ATTEMPTED (seed_only) | 33 | 33 | 0 |
| prove DONE (invented) | 1 | 2 | +1 |
| prove FAILED (invented) | 0 | 2 | +2 |
| proved-with-invented theorems | 1 | 2 | +1 |

**Every new prove attempt since resume targeted an invented-vocab task.**
The boost completely redirected the prover.

### Refined read: priority + depth

The 1 new proved theorem since resume — `self_dual_iff_double_dual_fixes`
— has the proof:

```lean
theorem self_dual_iff_double_dual_fixes {α : Type*} (M : Matroid α) :
    self_dual_matroid M ↔ M✶ = M := Iff.rfl
```

The proof is `Iff.rfl` because `self_dual_matroid` is *defined* as
`M✶ = M`. So this is a tautological restatement of a definition, not
substantive content.

The 2 failures (so far) are on substantive matroid statements —
`spanning_circuit_iff_base_insert`, `spanning_circuit_rank_identity`,
`spanning_circuit_symm_diff_base` — each exhausting 3 proof attempts
without success. These are real classical matroid results (a spanning
circuit has cardinality `rank + 1`; deleting any element gives a base).

So the corrected diagnosis:

- **Priority** was the dominant bottleneck — most invented-vocab tasks
  weren't even being tried. The boost fixes that.
- **Depth** is a residual bottleneck — even when attempted, substantive
  invented-vocab claims fail because the prover can't construct the
  multi-step matroid argument. The boost doesn't help with this.

The system can prove invented-vocab statements that have a 1-line
discharge (Iff.rfl, simp, aesop in trivial cases). It cannot prove
invented-vocab statements that require chaining several Mathlib lemmas
in a non-obvious way.

## Implications for "humanely meaningful" output

REPORT_MATROID.md ended with: *"the discovery system produces internally
consistent output that doesn't ground in shared vocabulary."* The
alignment + priority experiments add three sharpenings:

1. **Most invented vocabulary is not seed-aliased.** 4 of 110 testable
   (~4%) collapse back to seeds. The bulk is symbolically distinct
   (though many will likely Mathlib-align — a Phase 2 hypothesis to test).

2. **Of the proved corpus, 28/29 = 97% lives entirely in seed
   vocabulary.** The discovery system's *output* is dominated by
   restated seed-vocabulary results, even though its *proposals* (679
   conjectures, 33% using invented vocab) are much more diverse.

3. **The proposer-verifier gap survives re-prioritization.** Priority
   was a real bottleneck, but giving the verifier the floor only unlocks
   tautological restatements of definitions. The substantive
   invented-vocab content remains unreached. Reasonable reflection isn't
   just blocked by attention; it's blocked by proof-construction depth
   when the proof can't be a one-liner.

## What's left to try

- **Phase 2 alignment** (vs `canonical_matroid_mathlib.json`). Likely
  reclassifies a chunk of the 63 "novel" concepts as Mathlib-aliases,
  giving a sharper count of what's actually new content.
- **Invented-vs-invented alignment** for synonym clustering. Catches the
  multiple-loopless-variants case where no seed has the relevant shape.
- **Worth-tweak run completion** (in flight). The full 200-attempt
  resume tail will quantify the trivial-vs-substantive split among
  newly-proved invented-vocab theorems. If trivial dominates by 4:1 or
  more, "depth is the residual bottleneck" is firmly established.
- **Direct attack on depth**: increase `max_attempts` on `prove` and
  `proof_repair` tasks (currently 3); add a multi-step proof strategy
  (currently `direct_tactic_proof` and `structured_proof`); or
  goal-direct the prover toward a specific target theorem rather than
  letting it pick.

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

Lean-checking the LLM-invented matroid vocabulary against canonical
matroid vocabulary shows that ~96% of it is symbolically novel; the
discovery system's proved corpus largely sidesteps this novelty by
proving in seed vocabulary; and giving the prover priority to attempt
the invented-vocab pile unlocks tautological restatements of definitions
but not substantive proofs.
