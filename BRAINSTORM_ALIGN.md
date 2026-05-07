# Brainstorm — Alignment, Grounding, Namespace Awareness

A scratch doc for thinking through the "synonym tower" problem in the
matroid run and what a real fix would look like. No single
recommendation; the goal is to enumerate facets and tradeoffs so a
later design doc can pick one.

## The problem we're trying to solve

The matroid run produced **3 variants of "loop"** (`is_loop`,
`is_loop_specialized`, `loop_as_dual_coloop` — all α-equivalent to
`Matroid.IsLoop`) and **4 variants of "cocircuit"** (`is_cocircuit`,
`is_cocircuit_def`, `cocircuit_as_dual_circuit`,
`cocircuit_as_circuit_of_dual` — all aliasing to `Matroid.IsCocircuit`).
Plus several variants of seed concepts (`loop_element` redefining
`is_loop_def`, etc.). Roughly 5–10% of the invented namespace is
redundant.

This is "the synonym tower" failure mode. Symptoms:
- Wasted Bedrock + Lean compute generating theorems about duplicate concepts.
- Corpus illegibility — a mathematician reading the output can't tell
  if `is_loop_specialized` is "loop with extra hypothesis" or just a
  rename.
- Lost cross-cluster opportunities — theorems that should connect
  concepts under canonical names are split across synonyms.

## Diagnosis: it's a context-availability problem

When the discovery worker fires a heuristic, the LLM prompt contains:

- The target concept being expanded (1 concept).
- The target's `related_concepts` list (up to a few, mostly LLM-nominated
  in prior iterations).
- A few neighbors retrieved via the novelty index.

It does NOT contain:

- The full list of concepts already in the agenda (could be hundreds).
- The Mathlib namespace for the domain (`Matroid.IsLoop`,
  `Matroid.IsBase`, etc.).
- Any cross-domain canonical naming hint.

So when the LLM is told "expand the circuit concept," it has no way to
know `is_loop_def` already exists *as a concept name*. It either
invents a fresh name (`is_loop_specialized`, `loop_element`) or
re-derives the predicate from scratch under a different name. The
post-hoc alignment toolchain detects these duplicates but doesn't
prevent them.

## Facet 1: where to inject namespace awareness

Different injection points have different cost/latency/coverage profiles.

### A. In the proposal prompt (cheapest)

Pass the existing namespace to the LLM before it proposes. Two flavors:

- **Full namespace listing.** Compile `{name: (kind, 1-line desc)}` for
  all concepts in the domain, dump into the prompt. At 200 concepts
  this is ~16KB of prompt overhead — cheap relative to typical
  generation cost.
- **Embedding-retrieved subset.** Use the existing `NoveltyIndex`. For
  the current proposal context, retrieve top-K most-similar existing
  concepts and include them. Scales to large corpora; uses existing
  infrastructure. Risk: retrieval miss → LLM doesn't see the canonical
  predicate it should be reusing.

Trade-off: full listing is reliable until corpus grows huge; embedding
is scalable but lossy.

### B. At concept-creation time (post-LLM, pre-agenda)

The LLM has proposed a new concept. Before storing, run a check:

- **Cheap text similarity.** Compare the proposed body's tokens against
  every existing concept's body. If ≥X% similar, reject as duplicate.
  Catches `is_loop_specialized` whose body is byte-identical to
  `is_loop_def`. ~milliseconds.
- **Synchronous Lean alignment.** Run iff probes against compatible-shape
  existing concepts. Catches definitional duplicates with different
  body wording. ~75s per proposal — a 12× discovery slowdown.
- **Async Lean alignment.** Concept gets stored, but an alignment task
  fires in parallel. When it lands, retroactively flag the concept as
  alias and rewrite dependent theorems. Doesn't block; tower forms
  briefly then collapses.

Trade-off: text-similarity is fast but misses subtle aliases; sync Lean
catches everything but is too slow; async Lean is the compromise.

### C. At theorem-statement time

When a theorem is proposed using the new concept, check at *that* point
whether the concept aliases to canonical and rewrite the theorem to use
canonical names. Less invasive than rejecting the concept itself
(maybe the concept is fine but the theorem statement should be in
canonical form). Implementable as a post-processing step on every
proposed theorem.

### D. In the corpus, post-hoc

Run alignment after the discovery run completes. Walk the corpus,
rewrite all references to aliased concepts. Output a "canonicalized"
corpus.

Cheapest and easiest to build. Doesn't save energy during the run, but
gets the legibility benefit retroactively.

## Facet 2: what counts as "existing namespace"?

Three nested pools, each adding cost:

1. **Already-invented concepts in this run.** Cheap to enumerate
   (walk the agenda). Catches the worst of the synonym tower.
2. **Seed concepts.** Already accessible via the seed file. Catches
   re-inventions of seed predicates.
3. **Mathlib canonical predicates** in the domain's relevant modules
   (`Mathlib.Combinatorics.Matroid.*`). Curated list (we have
   `data/canonical_matroid_mathlib.json` with 12 entries). Catches
   re-inventions of textbook predicates.
4. **Wider Mathlib.** Predicates outside the domain that might be
   relevant (e.g., `Set.Finite`, `Set.Nonempty`). Probably overkill —
   too noisy, too domain-specific to enumerate.

Likely the right scope: (1) + (2) + (3). (4) is too speculative.

For matroid we've already curated (3); for other domains we'd need
similar curation. That itself is work.

## Facet 3: types of duplication

Not all "synonym tower" cases are the same:

- **Pure rename.** Body is byte-identical, just under a new name.
  Example: `loop_element` with body `def IsLoop ... := e ∈ M.E ∧ ¬ M.Indep {e}`
  is byte-identical to `is_loop_def`'s body. Cheapest to detect.
- **α-equivalent rename.** Bodies are different texts but
  definitionally the same after unfolding. Example: defining
  loopless as `∀ e ∈ M.E, M.Indep {e}` vs `∀ e ∈ M.E, ¬ is_loop_def M e`.
  Needs Lean to detect.
- **Specialization.** New concept is a *strict subset* of an existing
  one (extra hypothesis, narrower domain). Example: `is_loop_specialized`
  with extra ground-set hypothesis. Iff fails; needs `→` or `↔` with
  side condition. Probably should NOT be rejected — they're genuinely
  different concepts.
- **Generalization.** New concept is a *superset*. Same — genuinely
  different.
- **Conceptual variant.** Different definition, same intuitive name
  (e.g., "loop" defined via singleton-circuit vs via singleton-dependent).
  These are α-equivalent in matroid theory but not by definitional
  unfolding. Borderline case.

The first two are unambiguous duplicates. The middle two are not
duplicates. The last is a judgment call. A naive sync alignment check
catches the first two but might (or might not, depending on the tactic
ladder) catch the last.

## Facet 4: where does the LLM lose information?

Before assuming we need namespace injection, ask: does the LLM *already*
know about Mathlib's matroid module (it's been in Mathlib for years; the
LLM has seen it in training)? If yes, the synonym tower can't be a
"LLM doesn't know" problem at training-time level — it must be an
in-context "LLM doesn't see the right context" problem.

Empirical check: does the LLM, when prompted "what's the canonical name
for the predicate `e ∈ M.E ∧ ¬ M.Indep {e}` in Mathlib?", correctly
answer `Matroid.IsLoop`? If yes, the namespace fix is a reminder
problem; if no, it's a knowledge problem.

This matters because:
- If LLM knows but forgets → namespace prompt suffix is enough.
- If LLM doesn't know → we need to give it Mathlib's signatures or
  documentation in-prompt, which is bigger.

## Facet 5: cost/benefit numbers

Roughly, for a typical matroid-style discovery run:

| Approach | Per-proposal latency | Slowdown | % duplicates caught (est) |
|---|---|---|---|
| Prompt namespace (full) | +0s (just larger prompt) | 0.05× | 60–80% |
| Prompt namespace (embed retrieval) | +50ms (embedding lookup) | ~0.1× | 50–70% |
| Cheap text similarity post-LLM | +50ms | ~0.1× | 30–40% |
| Sync Lean alignment | +75s | 12× | ~95% |
| Async Lean alignment | +0s on critical path | 0× (background) | ~95% (delayed) |
| Post-hoc rewrite | +0s during run | 0× | 100% (after) |

"% caught" is a wild estimate. Real numbers need running experiments.

The clear winners on cost-effectiveness: prompt namespace (full) and
async Lean alignment, possibly stacked.

## Facet 6: interactions with the existing discovery loop

Several existing components touch the namespace:

- **Novelty check** (`discovery/checks/novelty.py`). Already gates
  proposals on cosine-distance to corpus. If a new concept is too close
  to an existing one, it's rejected. **But novelty operates on
  statements, not bodies, and uses a high threshold (0.92).** The 4
  loop variants weren't blocked because their statements had different
  surface text even though the bodies were identical.
  - Possible fix: add a body-similarity check parallel to the
    statement-similarity check, with a tighter threshold for
    definitional (`def`) concepts.
- **Deduplication by signature** (`discovery/__init__.py:
  is_duplicate_statement`). Hashes a normalized statement signature.
  Catches name-only renames if the signature matches. Doesn't catch
  body-level α-equivalence.
- **Heuristic worth** (`discovery/worth.py`). Boosts heuristics that
  produce admitted concepts. A heuristic that produces synonyms gets
  worth credit for each synonym, biasing toward duplicators.

Any namespace-awareness fix interacts with these: it should *not*
double-reject (novelty + namespace both saying "duplicate") and it
*should* feed back into worth (a heuristic that produces fewer
duplicates should be rewarded).

## Facet 7: prompts and instructions vs the LLM's intuition

There's an alternative framing: the LLM is told to invent concepts.
It's not told "if this concept already exists with a canonical name,
use that." So it dutifully invents. A prompt change could be as small
as:

> If a Mathlib predicate matches your proposed definition, use the
> Mathlib name and skip the rest of the definition. Common matroid
> predicates: `Matroid.IsLoop`, `Matroid.IsBase`, `Matroid.IsCircuit`,
> `Matroid.IsCocircuit`, `Matroid.IsHyperplane`, `Matroid.IsFlat`,
> `Matroid.Spanning`. Only invent a new name if the predicate doesn't
> match an existing one.

Cost: ~20 lines added to the system prompt. Latency: ~0. Coverage: hard
to predict; depends on whether the LLM follows the instruction.

This is the "tell the LLM not to do it" version. Worth trying as a
baseline before any infrastructure work.

## Facet 8: corpus-as-output vs corpus-as-internal-state

Two different goals:

1. **Internal corpus stays clean.** The system maintains canonical
   names internally; concepts are aliased at creation; theorems use
   canonical names from the start. Cleaner system; more invasive.
2. **External corpus is canonicalized.** The system can run with a
   messy internal namespace; a post-processing pass produces a clean
   external corpus for human readers. Simpler; doesn't fix the
   internal energy waste.

These imply different intervention points. (1) suggests in-loop
alignment (Facet 1.A or 1.B). (2) suggests post-hoc rewrite
(Facet 1.D).

For "humanely meaningful output" the user cares about (2). For the
system's self-improvement loop (distillation, etc.) (1) matters more.

## Facet 9: synonym tower as a feature, not a bug?

A counter-argument worth taking seriously: maybe synonyms are useful.
If the system invents `is_loop_specialized`, that's the system's way
of asking "what if this predicate had this extra hypothesis?" Even if
the predicate happens to coincide with `is_loop_def` once you simplify,
the *exploration step* of asking the question is part of how the
system explores the concept space.

Killing all synonyms might short-circuit useful exploration. The
question is: of the variants we found, were any of them generating
*different theorems* than the canonical concept would have? If yes,
the synonym was useful. If no, it was waste.

For the matroid run: the 3 loop variants seem to have generated
overlapping theorem sets. Probably mostly waste. But this is worth
checking before building infrastructure to kill them.

## Facet 10: what about cross-domain?

Everything above is matroid-specific. To make the fix general, the
canonical-Mathlib-pool curation needs to be per-domain (or at least
per-domain-cluster). For each new domain run we'd need:

- A seed file (already required).
- A canonical-Mathlib-pool file (currently only matroid has one).

This is real work. Could be partially automated by walking
Mathlib namespaces and extracting predicate signatures, but human
curation is probably needed to identify what's "canonical for this
domain."

## Open questions

- Does the LLM know `Matroid.IsLoop` exists at training time? (probe
  with a direct question)
- Of the 12 grounded concepts we found, were any of them prevented by
  the existing novelty index? (it should have caught body-identical
  cases — investigate why it didn't)
- For a sync alignment check at concept-creation time, is the per-Lean
  cost actually 75s, or could it be much lower with caching of
  Mathlib import overhead? (the `lake env lean` invocation per probe
  is doing a redundant Mathlib reimport every time)
- Is there a domain where the LLM's training-time exposure is so thin
  that namespace prompting won't help (because the LLM doesn't know
  the canonical names anyway)? Knot theory? Modal logic?
- For the matroid corpus we have, what fraction of the proved theorems
  would survive canonicalization without changes? With name rewrites?
  With body rewrites? This bounds how much canonicalization buys.

## Possible build order (not a recommendation)

If we were to spend time on this, plausible order:

1. **Probe**: ask the LLM directly "what's the canonical Mathlib name
   for `e ∈ M.E ∧ ¬ M.Indep {e}`?" Answer determines knowledge vs context
   problem.
2. **Cheapest**: add Mathlib-namespace nudge to the system prompt (Facet 7).
   Re-run matroid. Count synonyms in the output. Establishes baseline.
3. **Next cheapest**: add full-namespace listing of existing in-agenda
   concepts to the proposal prompt (Facet 1.A full). Re-run. Compare.
4. **Targeted**: tighten body-similarity check in novelty to catch
   byte-identical body duplicates (Facet 6, novelty interaction).
5. **Bigger**: switch to embedding-retrieved subset (Facet 1.A embed)
   once corpus grows.
6. **Backstop**: post-hoc canonicalization tool (Facet 1.D) for
   external corpus.
7. **Optional**: async in-loop alignment (Facet 1.B async) if there's
   value in catching the residue 6 doesn't.

## Summary

The synonym tower is mostly a context-availability problem, not an
alignment problem. The alignment toolchain we built detects duplicates
post-hoc. The deeper fix is to give the LLM enough context at proposal
time that it stops creating duplicates in the first place — starting
with the cheapest interventions (prompt-level namespace nudge) and
escalating only as needed (full-namespace listing → embedding retrieval
→ post-hoc canonicalization → async in-loop alignment).

Most of the remaining work is **prompt engineering and namespace
plumbing**, not new Lean infrastructure. The Lean-based alignment we
already have is the *verifier of last resort*, not the primary
mechanism.
