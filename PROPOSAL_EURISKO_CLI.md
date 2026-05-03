# Proposal: a one-shot CLI for the propose/check loop

A tight, single-iteration tool that runs `heuristic → propose → check → verdict`
end-to-end in 30-60 seconds. The development counterpart to the live agenda
system: same code paths, no scheduler, no waiting.

## 0. Why

The live system (agenda + scheduler + workers) is the *deployment* shape of
verified Eurisko. It runs continuously and produces the data the proposal is
about. It is, however, the wrong tool for *iterating on the design* of
heuristics, prompts, checkers, and gates.

The empirical problem from the modarith run:

- DiscoveryWorker turn: 10–15 min wall-clock when counterexample checks are on
  (5 fuel × 5 candidates × ~30s plausible probe each).
- ProofWorker turn: 5–10 min more.
- ReflectionWorker turn (if it claims a `reflect` task): another 30–60s.
- Phase 2 gate (`HeuristicSoundnessCheck`) only fires after all of the above
  chain through and Sonnet *also* decides to propose a new heuristic.

Net: the loop "does my heuristic prompt change actually shift admit rates?"
takes ~1 hour to close. That's not humanely viable for the kind of trial-and-
error required to discover *which* prompts and gates earn their keep.

The proposer/checker asymmetry only pays off if the human-in-the-loop can
iterate at human speed.

## 1. Goal

A one-shot CLI, `python run_one.py …`, that:

- Picks a heuristic and a target concept from a seed file.
- Runs the propose step (LLM call) once.
- For each candidate, runs the check pipeline (dedup → typecheck → novelty →
  counterexample) deterministically.
- Prints the full input/output of each step in a structured form.
- Exits in 30–60s on a cold cache, 5–15s on a warm cache.

Plus a sibling subcommand for Phase 2 specifically:

- `python check_soundness.py …` — runs `HeuristicSoundnessCheck` against a
  fixture of synthetic heuristic specs with known expected verdicts. Tests
  the *gate*, not the live system.

## 2. Non-goals

- **Not a replacement** for the agenda/scheduler/worker system. The live
  pipeline still produces the long-tail data; the CLI is for design work.
- **Not for batch runs.** If you want N concepts × M heuristics, you don't
  want this — you want the live system. The CLI is one shot.
- **Not distributed.** In-process only, single LLM call, single Lean compile.

## 3. Shape

```
python run_one.py \
  --domain modular_arithmetic \
  --seed data/seed_modular_arithmetic.json \
  --heuristic specialize \
  --concept mod_op \
  --num-conjectures 3 \
  --llm config/llm/aws-sonnet-4.5.yaml \
  [--no-novelty] [--no-counterexample] [--cache-dir .disco-cache]
```

Default `--llm` resolves the same way the scheduler config does (Hydra), so
the LLM choice matches the live run. Concrete model in the yaml.

Output is one structured block per candidate:

```
=== heuristic: specialize ===  template excerpt: "Add constraints to a definition..."
target concept: mod_op
prompt tokens: ~1200
LLM call: 3.2s — claude-sonnet-4-6

candidate 1: mod_op_self_zero
  statement: theorem mod_op_self_zero (n : Nat) : mod_op n n = 0
  dedup:        ok (no name/signature collision)
  typecheck:    PASS  (12.4s)
  novelty:      0.83 cosine to concept/modular_arithmetic/mod_op_self_zero  ← REJECT (≥ 0.92? no)
                ↳ admitted as novel
  counterexample: PASSED (slim_check ran 50 samples, no refutation)  (28.1s)
  → ADMIT  reason=typecheck_passed counterexample_check=passed novelty=0.83

candidate 2: ...
```

Plus a one-line summary at the end:

```
3 candidates: 2 admitted, 1 rejected (1× too_similar)  total: 47s
```

## 4. What gets reused (everything)

The CLI is a thin entry point over existing modules:

| step | module |
|---|---|
| seed loader | new — but trivially loads the same JSON the live system uses |
| heuristic selection | `discovery/heuristic.py` (Phase 2 dataclass) |
| propose: LLM call | `langchain_aws` / `langchain_google_vertexai` per Hydra `llm` config |
| parse output | `discovery.parse_conjecture_output` |
| dedup | `discovery.is_duplicate_statement` |
| typecheck | reuse `worker.discovery_worker._typecheck_conjecture` (extracted to a free function) |
| novelty | `discovery.checks.novelty.{NoveltyIndex, check_novelty}` |
| counterexample | `discovery.checks.counterexample.check_counterexample` |
| soundness test mode | `discovery.checks.soundness.check_heuristic_soundness` |
| LLM cache | new — see §5 |
| Lean cache | new — see §5 |

The point of this proposal is *not* to refactor or replace any of those. The
CLI just calls them in the right order with cheap inputs.

## 5. Caching layer

The single biggest win after eliminating the scheduler. Every step has an
input that's a deterministic function of (a few short strings + a config
hash), and the output is reproducible. Cache them all:

- **LLM cache**: key = `sha256(model_id + system_prompt + user_prompt +
  max_tokens)`. Value = the response text. Backed by a SQLite file
  (`.disco-cache/llm.sqlite`) so multiple processes can share. ~5ms hit.
- **Lean compile cache**: key = `sha256(source_text)`. Value = the
  `VerificationOutput` (outcome, status, stdout, stderr). Backed by a
  JSONL append-only file (`.disco-cache/lean.jsonl`) since Lean source
  hashes are stable and there's no need for SQL indexing. ~10ms hit.
- **Embedding cache**: already exists (`NoveltyIndex` saves a `.novelty.npz`).
  Extend to per-text-hash entries so single-statement queries don't rebuild
  the whole index.

All three caches are opt-in via `--cache-dir`. Default off → matches live
system behavior exactly. With cache on, second run of the same command is
near-instant; first run incurs only the steps that genuinely changed.

## 6. Soundness fixture mode

```
python check_soundness.py fixtures/synthetic_heuristics.jsonl
```

Where the fixture file is a JSONL of records like:

```json
{"name": "good_specialize_variant",
 "kind": "concept",
 "input_concept_kinds": ["operation", "definition"],
 "input_tags": [],
 "template": "SPECIALIZE the operation by ...",
 "expected_verdict": "passed",
 "harness_seed": "data/seed_modular_arithmetic.json"}

{"name": "bad_empty_template",
 "template": "do stuff",
 "expected_verdict": "empty_template",
 "harness_seed": "data/seed_modular_arithmetic.json"}
```

Output is a per-fixture row + summary:

```
fixtures/synthetic_heuristics.jsonl
  good_specialize_variant     PASS  (verdict=passed, expected=passed)
  bad_empty_template          PASS  (verdict=empty_template, expected=empty_template)
  bad_paraphrase_of_specialize PASS  (verdict=too_similar_to_existing, jaccard=0.91)
  bad_unmatched_filter        PASS  (verdict=applies_to_empty)
  ---
  4/4 passed in 0.8s
```

This is the unit-test-shaped version of the Phase 2 gate. Iteration on
`check_heuristic_soundness`'s logic happens against this in seconds, not
hours.

A few canonical fixtures land with this proposal so we can *demonstrate* the
gate's behavior even before the live system has produced a real reflection-
born heuristic. Once it does, we add more fixtures derived from real LLM
output and the test grows organically.

## 7. Phasing

### Phase A — basic run_one CLI (≈ half day)

1. `cli/__init__.py`, `cli/run_one.py`.
2. Argparse-style entry point with the flags in §3.
3. Hydra resolves `--llm` and `--seed`.
4. Extract `_typecheck_conjecture` and the post-typecheck pipeline from
   `DiscoveryWorker._create_concept` into free functions in
   `discovery/checks/__init__.py`. The worker continues to call them; the
   CLI also calls them.
5. Pretty-printed output as in §3.
6. No caching yet; every run is fresh.

Acceptance: run on `--heuristic specialize --concept mod_op` produces 1–5
candidates with full check verdicts in 30–60s, indistinguishable from what
the live system would do for the same inputs.

### Phase B — caching (≈ half day)

1. SQLite-backed LLM cache.
2. JSONL-backed Lean compile cache (or SQLite, whichever's nicer).
3. Per-text-hash embedding cache extension.
4. `--cache-dir` flag + `--no-cache` escape hatch.

Acceptance: re-running the same `run_one.py` command twice — second run
completes in <5s, and the trace shows "cache hit" lines for every step that
hadn't changed.

### Phase C — soundness fixture mode (≈ half day)

1. `cli/check_soundness.py`.
2. Fixture JSONL loader.
3. ~6 canonical fixtures covering each `verdict` value (`passed`,
   `empty_template`, `applies_to_empty`, `too_similar_to_existing`).
4. Pretty per-fixture output + total pass/fail summary.

Acceptance: fixtures all pass; modifying the threshold params in
`check_heuristic_soundness` and re-running shows the new verdicts within a
second.

## 8. What this changes about everything else

- **Phase 2 design loop closes in seconds**, not hours. The gate's behavior
  becomes something you can *tune empirically* against fixtures, then deploy
  to the live system once it's solid.
- **The reflection prompt becomes iterable**. `run_one --heuristic ... --reflect`
  (later subcommand) lets you inspect what Sonnet emits and immediately try a
  different prompt without restarting the scheduler.
- **The live system stays simple.** Nothing in the existing
  agenda/scheduler/worker code changes. The CLI is purely additive.
- **Distill data still flows.** The CLI invocations could write to the same
  trace JSONL format, so retrospective analysis works the same way as the
  live runs.

## 9. What NOT to do

- **Don't reimplement workers.** The CLI calls existing functions; if a
  worker has logic the CLI needs, factor it out — don't duplicate.
- **Don't introduce a new config system.** Stay on Hydra so `--llm` /
  `--seed` paths match the live system.
- **Don't add async** unless a step genuinely benefits. Synchronous code is
  easier to reason about and debug; the bottleneck is wall-clock per LLM
  call, not concurrency.
- **Don't bundle the CLI with the agenda.** Keep `cli/` and the live
  scheduler as separate entry points sharing the same library code under
  `discovery/`, `language/`, `worker/`. The CLI must work without ever
  constructing a `LocalAgenda`.

## 10. Open questions

- **Where does the CLI's output go?** Stdout for the first cut. Optional
  `--trace-jsonl <path>` later if we want machine-readable replay.
- **Reflection mode in the CLI.** Phase A leaves it out (focuses on
  Discovery's propose/check loop). Phase D could add it: load a (proved
  theorem, origin heuristic) pair, run reflection, parse the output, run
  the soundness gate on any proposed heuristic.
- **Multiple LLMs in one run.** A `--llm-discovery` and `--llm-proof` split
  is straightforward but probably not needed for Phase A.
- **Seed-less runs.** Phase A requires a seed file. Phase D could let you
  type a Lean statement directly: `python run_one.py --probe-statement "∀ n
  : Nat, n + 0 = n"` to test counterexample/novelty in isolation.
