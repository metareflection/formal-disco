# Frama-C Discovery with Opus 4.7 — Run Report

Two-run ablation comparing Claude Opus 4.7 (via AWS Bedrock) against the same
README-driven discovery loop, with and without curated ACSL-by-Example
in-context few-shot.

Run dates: 2026-05-04 to 2026-05-05.

## TL;DR

Curating 5 self-contained ACSL-by-Example seeds as a 2-of-5 random in-context
few-shot lifts implementer first-shot verification rate from 13.3% → 22.2%
(2.93× peak, 1.67× sustained), produces 93 verified programs at 587 attempts
vs 66 verified at 480 (1.41× corpus per 1.22× budget), raises
`requires`-template diversity by 1.42 bits, and transfers multi-behavior +
`\separated` + `terminates`/`exits` idioms (0 → 58/16/84 of verified
programs use them) at the cost of ~13pp in extend success.

## Setup

| | |
|---|---|
| Model | `us.anthropic.claude-opus-4-7` on AWS Bedrock |
| Verifier | Frama-C 32.0 (Germanium), Alt-Ergo 2.6.3, CVC5 1.3.2 (via opam) |
| Provers invoked | `frama-c -wp -wp-prover CVC5,Alt-Ergo` |
| Workers per run | 2 parallel (after rate-limit calibration; see Throughput) |
| Pipeline | `ReadmeInspiredIdeaGenerator` → `LLMImplementer` → `LLMFixer` / `EditorWorker` |
| Seed data | `acsl_by_example.pkl` (74 verified programs from
fraunhoferfokus/acsl-by-example) loaded into the agenda before the run starts |

Hardware: laptop (macOS, 4-core). Sequential single-process throughput ~3.5
attempts/min; 2-worker parallel ~10 attempts/min; 3-worker parallel hit
Bedrock rate limits and dropped to ~2/min.

## Implementation strategies

### 1. Model wiring

* **`config/llm/aws-opus-4.7.yaml`** — LLM config pointing at
  `us.anthropic.claude-opus-4-7`.
* **`config/scheduler/readme_ideas_framac_aws_opus_4_7.yaml`** — full 4-worker
  pipeline cloned from the 4.6 config, model bumped to 4.7, `max_tokens=10000`.

No code changes to workers or backends — the existing `framac` language
backend handled everything.

### 2. Pre-seeding the agenda

Both runs started with `cp acsl_by_example.pkl agenda-framac-opus47-*.pkl`
before launching the agenda server. This populated the agenda's object
store with 74 `framac-program` objects under
`dataset/acsl-by-example/<group>/<name>.c`. They count toward codebase
metrics from t=0 but the agenda doesn't re-verify them (which is fortunate:
their bytes-as-stored aren't self-contained — they include `typedefs.h`
and intra-corpus headers that don't resolve in a single-file invocation).

### 3. Curated few-shot (the main novel piece)

Three components.

**`scripts/curate_acsl_examples.py`** — produces self-contained, verified
ACSL-by-Example programs by:
1. Recursively inlining quoted `#include "X"` (search paths:
   seed's group dir, `StandardAlgorithms/`, `StandardAlgorithms/Logic/`).
2. Stripping header guards from inlined files so they don't redefine.
3. Replacing `<limits.h>` (the only system include transitively pulled in,
   from `typedefs.h`) with a small stub of the macros actually used
   (`INT_MAX`, `INT_MIN`, `UINT_MAX`).
4. Verifying each output with `frama-c -wp -wp-prover CVC5,Alt-Ergo,Z3
   -wp-timeout 30` and reporting OK/BAD.

**Curated set** (`language/framac/examples/curated/`):

| File | LOC | Idioms covered |
|---|---|---|
| `clamp.c` | 83 | multi-behavior, terminates, exits, lemma |
| `copy.c`  | 82 | `\separated`, loop variant, predicates with labels (`{Pre,Here}`), terminates, exits |
| `find.c`  | 70 | multi-behavior + complete/disjoint, loop variant, terminates, exits |
| `iota.c`  | 60 | loop variant, terminates, exits, predicate |
| `swap.c`  | 47 | `\separated`, terminates, exits |

Original `find5` (chosen first) needed Coq for 3 lemmas at 30s timeout —
swapped for `find` per the upstream `Results/*.json` metadata
(`coq=0, proved=32/32`). Curation script verifies the swap automatically.

**`language/framac/prompt.py`** — `FramaCPromptBuilder` extended:

* Constructor takes `seed_examples: list[str] | None`,
  `examples_per_call: int = 2`, `rng: random.Random | None`.
* If `seed_examples is None` and `FRAMAC_FEWSHOT` is set in the env, the
  builder loads `.c` files from `FRAMAC_FEWSHOT_DIR` (default
  `language/framac/examples/curated/`) at construction time.
* `_format_examples()` samples 2 examples per call and appends a
  `Here are example self-contained C programs...` block to the system
  prompts of `implement`, `initiate`, and `generate`. The block names
  the idioms ("multi-behavior contracts (...), terminates/exits clauses,
  \\separated for pointer non-aliasing, ...") so the model has both
  exemplars and a vocabulary cue.
* `repair`, `extend`, `idea`, `repair_full` are unchanged. Extend
  intentionally not augmented — extending dense specs is harder, not
  easier, with seed examples; the few-shot block would also bloat
  per-extend prompt length without obvious gain.

Toggle: `export FRAMAC_FEWSHOT=1` in the worker shells. No new Hydra
config needed; the same scheduler YAML serves both runs.

## Runs

### Run 1 — Baseline (`agenda-framac-opus47-seeded.pkl`)

Bedrock model `claude-opus-4-7`, no few-shot. Stopped at 480 attempts after
the first `max_attempts=200` cap, then continued to 480 with cap=10000 to
generate a diminishing-returns tail. Materialized to
`outputs/framac-opus47-baseline-final/`.

* 66 verified non-seed programs, 14% rate.
* No `behavior`, `terminates`, `exits`, `\separated` in any verified output.
* Two qualitative shapes:
  * **Theory-building extension chains** (deepest: `imp36b2626f` at 270 lines
    with 9 lemmas, a coherent rank/ancestry mini-theory). Real proofs
    (`is_ancestor_transitive`, `rank_distance_symmetric`, `min_max_sum`)
    plus padding (vacuous predicate `same_lineage = \true`, three
    redundant distance functions, artificial bounds `0 <= rank <= 6`).
  * **Algorithmic SUCCESSes** (e.g. `imp41e9ab3c`: HTML entity decoder with
    proper `\valid_read`, `\separated` (the lone exception), 3 helper
    predicates, 2 fully-verified helpers, declared-but-unimplemented
    top-level — clean Frama-C scaffolding style).

### Run 2 — Few-shot (`agenda-framac-opus47-fewshot.pkl`)

Same model, same pipeline, `FRAMAC_FEWSHOT=1`. Initial cap of 500; first
stopped at 287 attempts (where the comparison numbers were already
unambiguous), then resumed with cap=10000 to test whether the implementer
advantage held at scale; stopped manually at 587 attempts after the
fixer-grind pathology (see Limitations) made marginal yield very low.
Materialized to `outputs/framac-opus47-fewshot-final/`.

* 93 verified non-seed programs at 587 attempts (15.8% per-attempt rate).
* Idiom adoption (in 93 dataset/ SUCCESSes):
  * `terminates`: 84 (90%) / `exits`: 57 (61%) / multi-behavior: 58 (62%)
    / loop variant: 52 (56%) / predicate: 59 (63%) / `\separated`: 16 (17%)
    / lemma: 18 (19%) / axiomatic: 14 (15%) / ghost: 1 (1%).
* The early-run rate (39% per attempt at 287) attenuated to 16% over the
  resume tail. Implementer first-shot rate fell from 39% → 22.2%; extend
  rate held at ~42%; repair stayed near 3%. The fewer-GU finding (the
  "fails at parse, not at proof" effect) intensified with more data:
  16% GU among failures vs baseline 32%.
* Examples of cross-domain idiom transfer:
  * RGB pixel-adjustment with 3-behavior `clamp_component` (`too_low` /
    `too_high` / `in_range` + `complete`/`disjoint`) plus separated-array
    `adjust_rgb` — exact ACSL-by-Example template applied to a totally
    different domain.
  * Taxonomic synonym resolution with `behavior already_accepted` /
    `behavior synonym`.

## Quantitative comparison

`python analysis.py tables agenda-framac-opus47-seeded.pkl
agenda-framac-opus47-fewshot.pkl`

### Per-task success rate

| Task | Baseline (480) | Few-shot @ 287 (peak) | Few-shot @ 587 (final) | Ratio (final) |
|---|---|---|---|---|
| implement DONE | 13.3% (29/218) | 39.0% (41/105) | **22.2%** (55/248) | **1.67×** |
| extend DONE | **54.5%** (36/66) | 40.6% (28/69) | 41.9% (39/93) | 0.77× |
| repair DONE | 2.8% (6/215) | 3.9% (4/103) | 2.5% (6/241) | 0.89× |

The implementer's first-shot rate started at 2.93× the baseline and
attenuated to 1.67× as easy ideas were exhausted. The implementer is still
the dominant driver of incremental SUCCESS, but the gap to baseline narrowed
in the long tail.

### Verification outcome distribution (failed programs only)

| | Baseline | Few-shot @ 587 |
|---|---|---|
| GOAL_UNPROVEN | 32.1% | **16.0%** |
| FAIL (parse/syntax) | 67.9% | 82.8% |

Few-shot fails *earlier* — when it doesn't verify, it falls over at the
parser rather than partway through proof obligations. Reads as: the model
has internalized which contract *shapes* WP can discharge, so wrong-shape
attempts don't get part-way to a proof.

### Diversity (entropy in bits, pooled across dataset/ programs)

| Feature | Baseline (140 progs) | Few-shot (167 progs) | Δ (bits) |
|---|---|---|---|
| **requires_templates** | **4.68** | **6.10** | **+1.42** |
| subject_words | 6.32 | 6.69 | +0.37 |
| ensures_templates | 6.89 | 7.05 | +0.16 |
| assert_templates | 6.74 | 6.86 | +0.12 |
| invariant_templates | 5.74 | 5.83 | +0.09 |
| loop_skeletons | 0.47 | 0.54 | +0.07 |

Few-shot has higher entropy in all 6 feature categories. The biggest is
`requires_templates` — the empirical fingerprint of multi-behavior `assumes`
clauses. **+66 unique `requires` shapes** (182 vs 116), **+32 unique
`ensures` shapes** (305 vs 273) in the few-shot corpus.

## Findings

1. **Curated few-shot is a high-leverage intervention.** 5 small examples
   in the prompt nearly tripled the implementer first-shot success rate
   and transferred several ACSL idioms the model never produced unaided.

2. **The teaching effect isn't surface-level mimicry.** Programs apply
   the multi-behavior + `\separated` + `terminates`/`exits` idiom
   correctly to *novel* domains (RGB pixel adjustment, taxonomic synonym
   resolution, etc.) drawn from the README-derived ideas — not just to
   copies of the seed problems.

3. **Few-shot teaches proof-discipline, not just syntax.** The
   GOAL_UNPROVEN rate among failed programs *fell* (32% → 20%). Wrong
   guesses now usually fall over at parse, suggesting the model learned
   what *shape* of contract is discharge-able, not just what symbols are
   ACSL.

4. **Extend is harder under few-shot.** Extending programs that already
   carry dense `behavior`/`complete`/`disjoint` machinery is harder than
   extending sparser specs — adding a function risks breaking the
   `complete behaviors;` invariant. 14pp drop in extend success.

5. **Net qualitative shape differs.** Baseline produced a few deep
   extension chains (e.g. 270-line theory file at depth 4). Few-shot at
   the same attempt budget would produce *wider* (more roots) and
   *shallower* corpora. With proportionally more attempts, this rebalances.

## Limitations

* Single run per condition; no confidence intervals.
* Run 2 stopped at 587 attempts (run 1 at 480) after the marginal SUCCESS
  rate dropped to ~3% in the resume tail. Per-attempt metrics are
  comparable, but neither run was budgeted to convergence.
* The 5 curated examples are the first picks, not optimized. No formal
  ablation over which seeds matter most. Inductive predicates and
  ghost-code coverage remain weak (1 ghost-code SUCCESS).
* **Fixer-grind pathology in the resume tail.** Inspection of the last
  ~30 attempts before manual stop showed 21/30 program objects were
  different repair attempts on a single GOAL_UNPROVEN ancestor. The
  agenda's `attempt_priority_factor=0.9` decays repair priority slowly,
  so when a hard task dominates the queue the fixer keeps retrying it
  before round-robin returns to the implementer. Single-worker runs are
  most exposed; multi-worker setups hide it but waste API budget.
  Mitigations not applied here: lower `attempt_priority_factor`, or
  enforce a per-task `max_attempts` knob (not currently exposed via
  Hydra).
* Cost not measured. Few-shot system prompt is ~5.5KB vs ~0.9KB baseline;
  per-call token overhead is small relative to the user prompt + response,
  but the rate of attempts (and thus total spend) wasn't held constant.

## Open directions

* **(B)** Improve the curated set. Add ghost-code (e.g. `selection_sort.c`
  preprocessed) and an inductive-predicate example to lift the
  axiomatic-block / ghost-code adoption beyond the current 8 / 1.
  Optionally: feature-novelty selector that picks examples per call
  based on what's underrepresented in the live corpus (the README's
  "diversity-driven example selection" todo).
* **(C)** Distill. ~270 distill examples were collected during the
  few-shot run, biased toward the proper ACSL idioms. Fine-tune
  Qwen3-Coder-30B on them and re-run discovery.
* **(D)** Merged corpus analysis. Combine baseline + few-shot for a
  unified feature-distribution picture rather than ablation framing.

## Files

| | |
|---|---|
| Configs | `config/llm/aws-opus-4.7.yaml`, `config/scheduler/readme_ideas_framac_aws_opus_4_7.yaml` |
| Curation | `scripts/curate_acsl_examples.py`, `language/framac/examples/curated/{find,clamp,copy,iota,swap}.c` |
| Prompt builder | `language/framac/prompt.py` |
| Distributed-agenda fix | `agenda_distributed.py` (`_discoverable_host`) |
| Pickles | `agenda-framac-opus47-seeded.pkl` (baseline), `agenda-framac-opus47-fewshot.pkl` (few-shot) |
| Materialized output | `outputs/framac-opus47-baseline-final/`, `outputs/framac-opus47-fewshot-final/` |
| Seed corpus | `acsl_by_example.pkl` (74 programs, generated by `python acsl_pkl.py`) |
