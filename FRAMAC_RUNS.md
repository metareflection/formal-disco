# Frama-C Discovery on Open Models — Two-Run Report

Single-node slurm runs comparing a stock open model against an SFT-distilled
variant trained on the curated few-shot Opus 4.7 corpus from
[FRAMAC_REPORT.md](FRAMAC_REPORT.md). Both pickles are in `agendas-framac/`.

## TL;DR

SFT-distilling Qwen2.5-Coder-32B on the 93-program curated few-shot Opus 4.7
agenda lifts dataset throughput from **355 to 6174 verified programs** at
identical 100k-attempt budgets — a **17.4× corpus** under the same scheduler,
workers, and Frama-C stack. The lift is dominated by the implementer
(0.5% → 13.4%, 25.4×) and the resulting downstream extend chain
(0.3% → 24.4% on 9.6× more extend attempts). Distilled programs are
simpler on average (body size 4.45 → 2.40), but contribute 8.7× more
unique `requires` and 4.2× more unique `ensures` shapes in absolute terms.

## Setup

Both runs share `slurm/single-node-framac.sbatch` (commit `f15f67d`) on
`seas_gpu`: 2× GPU, 16 CPU, 300 G RAM, 32 vLLM-served workers, scheduler
`config/scheduler/disco_framac_vllm.yaml` (Initiator + LLMFixer +
EditorWorker, `turn_fuel=5`, `attempt_priority_factor=0.9`). vLLM serves
the model at `max-model-len=8K`, `tensor-parallel-size=2`. Verifier:
Frama-C 32.0 with Alt-Ergo + CVC5 over Why3. Agenda budget capped at
`max_attempts=100000`.

| | Run 1 (baseline) | Run 2 (few-shot SFT) |
|---|---|---|
| Run name | `sn-framac__qwen3-coder-30b` | `sn-framac__opus-fewshot-sft` |
| Pickle | `agenda-sn-framac__qwen3-coder-30b.pkl` | `agenda-sn-framac__opus-fewshot-sft.pkl` |
| Served model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | LoRA-merged `Qwen2.5-Coder-32B-Instruct` |
| SFT data | — | `agenda-framac-opus47-fewshot.pkl` (success-only, 93 verified) |
| SFT recipe | — | `config/distill.yaml`: r=32, α=64, 3 epochs, lr 2e-4, max_seq=8192 |
| Order | first | **second** |

The few-shot-SFT model is the Open-Direction (C) from FRAMAC_REPORT.md:
distill the Opus 4.7 + curated few-shot agenda, fine-tune an open model,
re-run discovery from scratch.

## Quantitative comparison

`python analysis.py tables agendas-framac/agenda-sn-framac__qwen3-coder-30b.pkl agendas-framac/agenda-sn-framac__opus-fewshot-sft.pkl`

### Attempt budget split

| Task type | Baseline | Few-shot SFT |
|---|---|---|
| `initiate-program` | 48,456 (48.5%) | 34,805 (34.8%) |
| `repair` | 48,376 (48.4%) | 34,732 (34.7%) |
| `extend` | 3,168 (3.2%) | **30,463 (30.5%)** |
| Total | 100,000 | 100,000 |

The scheduler's round-robin allocates extend only when there is an existing
verified program to extend from. Baseline's near-zero extend share is a
consequence of its low initiate yield, not a scheduler difference.

### Per-task success rate (DONE / total)

| Task | Baseline | Few-shot SFT | Ratio |
|---|---|---|---|
| `initiate-program` DONE | 0.5% (259 / 48,467) | **13.4%** (4,661 / 34,819) | **25.4×** |
| `extend` DONE | 0.3% (1 / 355) | **24.4%** (1,504 / 6,174) | **81.0×** |
| `repair` DONE | 0.2% (96 / 48,550) | 0.4% (125 / 31,433) | 2.0× |

Repair is the slowest-moving lever: baseline and SFT both attempt many
repairs, but on the SFT side the implementer/extend pipeline keeps the
dataset queue full enough that repair stays a tail.

### Verified-program throughput

| Metric | Baseline | Few-shot SFT |
|---|---|---|
| Verified programs in `dataset/` | 355 | **6,174** |
| SUCCESS verification outcomes (raw) | 2 | 3,497 |
| GOAL_UNPROVEN outcomes (raw) | 135 | 45 |
| FAIL outcomes (raw) | 48,319 | 31,263 |

Among failed programs, GOAL_UNPROVEN share drops from 73.6% (135/183) to
0.14% (45/31,308). On the open-model side the few-shot SFT's failures are
now overwhelmingly parse/typecheck failures, not stuck WP obligations —
the same "wrong-shape attempts don't get part-way to a proof" effect
described in FRAMAC_REPORT.md for the in-context few-shot, but
much more pronounced here.

### Complexity (per-program means over `dataset/`)

| Metric | Baseline | Few-shot SFT |
|---|---|---|
| `body_sizes` | 4.45 | 2.40 |
| `n_idents_in_asserts` | 0.08 | 0.03 |
| `n_idents_in_invs` | 0.56 | 0.06 |
| `n_loops_per_fn` | 0.17 | 0.04 |

SFT programs are systematically smaller and contain fewer loops and
invariant identifiers. Read as: the SFT model converges to shallow,
many-leaf programs (chain depth one to two from an initial implement),
whereas the baseline's rare successes are slightly meatier on average
but vastly fewer.

### Diversity (entropy in bits, pooled across `dataset/`)

| Feature | Baseline | Few-shot SFT | Δ (bits) |
|---|---|---|---|
| `subject_words` | 7.48 | **8.75** | +1.27 |
| `loop_skeletons` | 1.15 | **1.60** | +0.45 |
| `assert_templates` | 3.61 | **3.90** | +0.29 |
| `ensures_templates` | **6.59** | 6.26 | −0.33 |
| `invariant_templates` | **3.72** | 3.55 | −0.17 |
| `requires_templates` | **5.55** | 4.10 | −1.45 |

Per-program entropy is mixed: the SFT corpus is more diverse in subject
words and loop skeletons, less diverse in `requires` / `ensures` /
invariant templates. The latter reflects mode-collapse onto the seed
idioms — many programs reuse the multi-behavior `requires \valid(...)` /
`requires <bound>` patterns from the curated examples.

### Diversity (unique features in absolute counts)

| Feature | Baseline | Few-shot SFT | Ratio |
|---|---|---|---|
| `subject_words` | 530 | 1,873 | 3.5× |
| `ensures_templates` | 395 | 1,673 | 4.2× |
| `requires_templates` | 242 | 473 | 2.0× |
| `assert_templates` | 20 | 56 | 2.8× |
| `invariant_templates` | 52 | 58 | 1.1× |
| `loop_skeletons` | 9 | 18 | 2.0× |

In absolute terms SFT produces ~17× more programs but only ~2–4× more
unique templates — consistent with per-program entropy being lower while
total corpus coverage is broader. The biggest gap is `ensures_templates`
(395 → 1,673), where the few-shot SFT has internalized a wide library
of postcondition idioms.

## Findings

1. **Cross-model transfer of curated few-shot works.** Idioms taught to
   Opus 4.7 via 5 in-context ACSL-by-Example seeds, distilled through
   its 93-success agenda, transfer to a 32B open model and produce a
   17× corpus uplift on a different base architecture
   (Qwen2.5-Coder-32B vs. the Qwen3-Coder-30B baseline).

2. **Implementer first, extend follows.** The 25× implementer lift is
   the load-bearing change. Extend's apparent 81× lift is partly a
   second-order effect: with a real implementer, the scheduler can
   actually allocate extend turns and chain depth becomes possible.

3. **Repair stays cheap-and-shallow.** Both runs show ~0.2–0.4% repair
   DONE rates. Repair on bad initiates remains low-yield on open
   models; the gains come from getting the initial implement right.

4. **GOAL_UNPROVEN nearly vanishes for SFT.** Distilled failures parse
   but use unsound contracts very rarely (45 in 100k vs 135). The
   model has learned which contract shapes WP can discharge.

5. **Shallow-wide vs deep-narrow regime.** Body sizes 2.40 vs 4.45 and
   the absolute-vs-entropy diversity split together say: the few-shot
   SFT produces a wide flat carpet of small verified programs; the
   baseline's rare successes are individually slightly larger and use
   slightly more elaborate ensures/invariant patterns per program.

## Limitations

* Different base models. Baseline is Qwen3-Coder-30B-A3B-Instruct, the
  SFT model is LoRA-merged Qwen2.5-Coder-32B-Instruct. The comparison
  is therefore "open-model status quo vs. SFT-on-distilled-corpus", not
  a clean SFT ablation against the same base. A proper ablation
  (SFT-vs-base on the same Qwen2.5 backbone) is not run here.
* Single run per condition; no confidence intervals.
* SFT data was capped at success-only (`success_only: true`,
  `treat_goal_unproven_as_success: false`) and trained for 3 epochs at
  rank 32 — no hyperparameter sweep.
* Both runs used the same 100k-attempt cap, which the SFT model uses
  much more efficiently. Holding compute constant (e.g. matching
  verified-program counts) would change the comparison shape.
* No cross-run dataset deduplication: the corpus-size advantage may
  inflate raw unique-feature counts via near-duplicate programs that
  differ only in identifier renames.

## Files

| | |
|---|---|
| Slurm | `slurm/single-node-framac.sbatch` |
| Scheduler | `config/scheduler/disco_framac_vllm.yaml` |
| SFT config | `config/distill.yaml` |
| Baseline pickle | `agendas-framac/agenda-sn-framac__qwen3-coder-30b.pkl` |
| Few-shot SFT pickle | `agendas-framac/agenda-sn-framac__opus-fewshot-sft.pkl` |
| Source few-shot agenda | `agenda-framac-opus47-fewshot.pkl` (from FRAMAC_REPORT.md run 2) |
| Analysis | `python analysis.py tables <baseline.pkl> <fewshot.pkl>` |
