# Eurisko Natural Numbers Experiment — Status

## Experiment

Seeded the discovery system with 15 natural number concepts (7 operations, 3
definitions, 5 basic theorems about zero/successor/addition) and let it run
to see what it discovers and proves.

Config: `discovery_nat_vertex.yaml`, Sonnet for discovery/reflection, Opus for
proofs/repair, Lean 4 + Mathlib verification.

## Results

**Discovery works.** The system generated 309 concepts from 15 seeds:
- 62 new definitions (e.g., `positive_nat`, `even_nat`, `BoundedSucc`, `SuccClosedSet`, `StrictlyMonotoneNat`)
- 237 conjectures spanning successor, addition, multiplication, exponentiation, divisibility, parity, ordering
- 1 instance

Heuristic worth tracking is functioning — `algebraic_identities` rose to the
top (3/10 success rate, interest 2.20), while `boundary_cases` generated many
conjectures but none proved (0/19). Reflection created weakened conjectures
from failures.

**Proof is the bottleneck.** Of 237 conjectures, only 7 were proved — all via
embedded proofs from the discovery phase, none via the LLM proof strategies:
- `direct_tactic_proof`: 0/14
- `structured_proof`: 0/13
- Embedded (Sonnet's inline proof suggestions): 7 verified out of ~170 that had them

The proved theorems are all basic successor properties (injectivity, no fixed
points, succ ≠ zero, etc.). The more interesting conjectures about addition
commutativity, multiplication distributivity, and exponent laws sit unproved
in the queue.

## What works

1. **Concept generation**: The heuristics (specialize, generalize,
   compose_operations, algebraic_identities, boundary_cases, analogy_transfer)
   produce a diverse set of definitions and conjectures from minimal seeds.

2. **Heuristic worth evolution**: Success rates drive exploration. Heuristics
   that produce provable conjectures get boosted; those that don't get demoted.

3. **Deduplication**: Statement-signature dedup (via `_sig/` objects) prevents
   literal duplicate conjectures (same statement, different names).

4. **Embedded proof shortcut**: When the discovery LLM happens to produce a
   correct proof inline, the proof worker verifies it directly without an
   expensive Opus call.

5. **Library output**: Proved theorems are written to `LeanDisco/Domains/NaturalNumbers/`
   as self-contained `.lean` files.

## What doesn't work

### 1. LLM proof generation (0% success rate)

The proof worker sends the theorem statement + strategy hint to Opus and gets
back Lean code that never verifies. Root causes:

- **No error feedback loop.** The proof worker tries a proof, it fails, and
  it tries a completely new proof from scratch. It never shows the LLM the
  Lean error message from the failed attempt. The `LLMFixer` worker exists
  for exactly this purpose but the proof worker doesn't create `repair` tasks
  for its own failures — it only creates `reflect` tasks after all attempts
  are exhausted.

  **Fix:** On each failed proof attempt, create a `repair` task for `LLMFixer`
  with the failed code and the Lean error output. This gives the fixer a
  chance to iterate on the specific error rather than starting from scratch.

- **Lean errors go to stdout, not stderr.** The backend captures both but
  returns `stderr` in the `VerificationOutput`. Workers logging errors see
  empty stderr. Not a blocking issue but makes debugging harder.

### 2. Missing definitions in proof files

Conjectures reference custom definitions (`add_op`, `succ_op`, etc.) that
exist only in the agenda, not in Mathlib. The proof file must include them.
The current approach gathers definitions from `related_concepts`, but:

- `related_concepts` doesn't always list every definition the statement
  references. A statement using `add_op` might only list `mul_op` as related.
- **Fix:** Scan the statement text for known concept names and include their
  definitions, rather than relying solely on `related_concepts`.

### 3. Malformed statements from the discovery LLM

The discovery LLM (Sonnet) sometimes wraps statements in backticks, prepends
natural language descriptions, or generates bare propositions without
`theorem` keywords. We added `_strip_code_fences` cleaning but ~6 out of 318
conjectures still have issues. The typechecker should catch these before they
enter the prove queue, but some slip through.

### 4. Round-robin scheduling vs. proof throughput

The round-robin scheduler gives equal fuel to all workers. Discovery (Sonnet
calls) is fast; proof verification (Lean compilation) takes ~15-30s per
check; LLM proof generation (Opus calls) takes 30-120s. The queue grows much
faster than it drains.

We worked around this by not counting embedded proof successes against fuel,
and deferring LLM attempts on first failure. But these are hacks on top of
a scheduler that doesn't distinguish cheap work from expensive work.

### 5. Deduplication is ad hoc

Statement-signature dedup uses `_sig/{domain}/{hash}` sentinel objects — a
shadow index bolted onto an object store that only supports path-based lookup.
Works but architecturally wrong. The proper fix is to add content-based query
support to the Agenda, or use Lean-level definitional equality checking for
semantic dedup.

## Recommendations

1. **Connect proof failures to the fixer.** This is the highest-leverage
   change. The fixer already exists and handles error-driven repair. The proof
   worker just needs to create `repair` tasks with the failed code + error
   output instead of (or in addition to) retrying from scratch.

2. **Broader definition inclusion.** Grep the statement for known definition
   names rather than relying on `related_concepts`. This fixes the missing-def
   problem that causes many embedded proofs to fail.

3. **Separate embedded verification from LLM proving.** Instead of the current
   interleaved logic in `ProofWorker._process_task`, run a fast embedded-proof
   sweep as a distinct phase (or worker), then queue remaining conjectures for
   LLM proof attempts. Cleaner than the current conditional pile-up.

4. **Priority-aware scheduling.** Let the scheduler weight worker turns by
   queue depth or estimated cost, rather than equal round-robin. Or let
   workers declare their own fuel consumption model.

## Files changed for this experiment

- `data/seed_natural_numbers.json` — 15 seed concepts
- `config/scheduler/discovery_nat_vertex.yaml` — scheduler config
- `discovery/__init__.py` — statement dedup, related_concepts sanitization,
  improved `_strip_code_fences`
- `discovery/prompts.py` — (reverted, no net change)
- `worker/proof_worker.py` — embedded proof shortcut, fuel-free embedded,
  deferred LLM, preamble assembly
- `worker/concept_seeder.py` — (no change, parameterized by config)
- `worker/discovery_worker.py` — statement dedup check, import of
  `is_duplicate_statement`
- `worker/reflection_worker.py` — statement dedup check
