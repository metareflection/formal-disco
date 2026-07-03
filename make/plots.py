"""Plot targets that shell out to analysis.py."""

import subprocess

from . import target


AGENDAS_DIR = "agendas"
DAFNYBENCH_DIR = "data/DafnyBench/DafnyBench/dataset/ground_truth/"

DIVERSITY_SUFFIXES = ["entropy", "unique", "pnew", "pnew-program"]


def _diversity_outputs(name: str) -> list[str]:
    return [
        f"plots/{name}-{suffix}.{ext}"
        for suffix in DIVERSITY_SUFFIXES for ext in ("svg", "png")
    ]


def _outputs(name: str) -> list[str]:
    return [f"plots/{name}.svg", f"plots/{name}.png"]


def _run(*args: str) -> None:
    subprocess.run(["python", "analysis.py", *args], check=True)


# ----- Comparison 1: GitHub READMEs (with vs without) -------------------------

GH_AGENDAS = [
    f"{AGENDAS_DIR}/agenda-sn-monolithic__gh__claude-opus.pkl",
    f"{AGENDAS_DIR}/agenda-sn-monolithic__nogh__claude-opus.pkl",
    f"{AGENDAS_DIR}/agenda-sn-monolithic__gh__claude-sonnet.pkl",
    f"{AGENDAS_DIR}/agenda-sn-monolithic__nogh__claude-sonnet.pkl",
]
GH_LABELS = "Mono+GH (Opus),Mono (Opus),Mono+GH (Sonnet),Mono (Sonnet)"


@target(name="compare-gh-task-success-rate",
        generates=_outputs("compare-gh-task-success-rate"))
def _gh_success() -> None:
    _run("plot-task-success-rates", *GH_AGENDAS,
         "--labels", GH_LABELS, "--name", "compare-gh-task-success-rate")


@target(name="compare-gh-program-complexity",
        generates=_outputs("compare-gh-program-complexity"))
def _gh_complexity() -> None:
    _run("plot-program-complexity", *GH_AGENDAS,
         "--labels", GH_LABELS, "--name", "compare-gh-program-complexity",
         "--dafnybench", DAFNYBENCH_DIR, "--x-clip-percentile", "90")


@target(name="compare-gh-program-complexity-absolute",
        generates=_outputs("compare-gh-program-complexity-absolute"))
def _gh_complexity_absolute() -> None:
    _run("plot-program-complexity", *GH_AGENDAS,
         "--labels", GH_LABELS, "--name", "compare-gh-program-complexity",
         "--dafnybench", DAFNYBENCH_DIR, "--absolute",
         "--x-clip-percentile", "90", "--lower-percentile", "25")


@target(name="compare-gh-program-diversity",
        generates=_diversity_outputs("compare-gh-program-diversity"))
def _gh_diversity() -> None:
    _run("plot-program-diversity", *GH_AGENDAS,
         "--labels", GH_LABELS, "--name", "compare-gh-program-diversity")


# ----- Comparison 2: Monolithic vs Disco3 -------------------------------------

DISCO3_AGENDAS = [
    f"{AGENDAS_DIR}/agenda-sn-monolithic__gh__claude-opus.pkl",
    f"{AGENDAS_DIR}/agenda-sn-disco3_claude-opus.pkl",
    f"{AGENDAS_DIR}/agenda-sn-monolithic__gh__claude-sonnet.pkl",
    f"{AGENDAS_DIR}/agenda-sn-disco3_claude-sonnet.pkl",
]
DISCO3_LABELS = "Mono+GH (Opus),Disco3 (Opus),Mono+GH (Sonnet),Disco3 (Sonnet)"


@target(name="compare-disco3-task-success-rate",
        generates=_outputs("compare-disco3-task-success-rate"))
def _disco3_success() -> None:
    _run("plot-task-success-rates", *DISCO3_AGENDAS,
         "--labels", DISCO3_LABELS, "--name", "compare-disco3-task-success-rate")


@target(name="compare-disco3-program-complexity",
        generates=_outputs("compare-disco3-program-complexity"))
def _disco3_complexity() -> None:
    _run("plot-program-complexity", *DISCO3_AGENDAS,
         "--labels", DISCO3_LABELS, "--name", "compare-disco3-program-complexity",
         "--dafnybench", DAFNYBENCH_DIR, "--x-clip-percentile", "90")


@target(name="compare-disco3-program-complexity-absolute",
        generates=_outputs("compare-disco3-program-complexity-absolute"))
def _disco3_complexity_absolute() -> None:
    _run("plot-program-complexity", *DISCO3_AGENDAS,
         "--labels", DISCO3_LABELS, "--name", "compare-disco3-program-complexity",
         "--dafnybench", DAFNYBENCH_DIR, "--absolute",
         "--x-clip-percentile", "90", "--lower-percentile", "25")


@target(name="compare-disco3-program-diversity",
        generates=_diversity_outputs("compare-disco3-program-diversity"))
def _disco3_diversity() -> None:
    _run("plot-program-diversity", *DISCO3_AGENDAS,
         "--labels", DISCO3_LABELS, "--name", "compare-disco3-program-diversity")


# ----- Comparison 3: Distillation + self-improvement --------------------------

DISTIL_AGENDAS = [
    f"{AGENDAS_DIR}/agenda-sn-disco3_claude-opus.pkl",
    f"{AGENDAS_DIR}/agenda-sn-disco3__qwen2.5-coder-32B__100k.pkl",
    f"{AGENDAS_DIR}/agenda-sn-disco3__qwen2.5-coder-32B_opus-distilled.pkl",
    f"{AGENDAS_DIR}/agenda-sn-disco3__qwen2.5-coder-32B_opus-distilled_si1.pkl",
    f"{AGENDAS_DIR}/agenda-sn-disco3__qwen2.5-coder-32B_opus-distilled_si2.pkl",
]
DISTIL_LABELS = ("Claude Opus,Qwen2.5 32B,Qwen2.5 32B (distil),"
                 "Qwen2.5 32B (distil+SI1),Qwen2.5 32B (distil+SI2)")


@target(name="compare-distil-task-success-rate",
        generates=_outputs("compare-distil-task-success-rate"))
def _distil_success() -> None:
    _run("plot-task-success-rates", *DISTIL_AGENDAS,
         "--labels", DISTIL_LABELS, "--name", "compare-distil-task-success-rate")


@target(name="compare-distil-program-complexity",
        generates=_outputs("compare-distil-program-complexity"))
def _distil_complexity() -> None:
    _run("plot-program-complexity", *DISTIL_AGENDAS,
         "--labels", DISTIL_LABELS, "--name", "compare-distil-program-complexity",
         "--dafnybench", DAFNYBENCH_DIR, "--x-clip-percentile", "90")


@target(name="compare-distil-program-complexity-absolute",
        generates=_outputs("compare-distil-program-complexity-absolute"))
def _distil_complexity_absolute() -> None:
    _run("plot-program-complexity", *DISTIL_AGENDAS,
         "--labels", DISTIL_LABELS, "--name", "compare-distil-program-complexity",
         "--dafnybench", DAFNYBENCH_DIR, "--absolute",
         "--x-clip-percentile", "90", "--lower-percentile", "50")

@target(name="compare-distil-program-diversity",
        generates=_diversity_outputs("compare-distil-program-diversity"))
def _distil_diversity() -> None:
    _run("plot-program-diversity", *DISTIL_AGENDAS,
         "--labels", DISTIL_LABELS, "--name", "compare-distil-program-diversity")


# ----- Rarefaction curves (feature-entropy vs. #programs) ---------------------

FINAL_AGENDAS = "final-runs/agendas"
VERUS_BENCH = "data/verus-proof-synthesis/benchmarks/Verus-Bench"
VERUSAGE = "data/verus-proof-synthesis/benchmarks/VeruSAGE-Bench"
SAFE_DIR = "data/external/safe"        # microsoft/Verus_Training_Data sft_safe_25k.json
VERUSYN_DIR = "data/external/verusyn"  # microsoft/Verus_Training_Data sft_part1_6.9M.json (first 100k)

RAREFACTION_CACHE = "plots/.rarefaction-cache"

# The two feature groups: features1 (the categorical/structural metrics shown in
# the main paper) and features2 (numeric size metrics, for the appendix).
RAREFACTION_FEATURES1 = "language_features,loop_skeleton,annotation_template,subject_word"
RAREFACTION_FEATURES2 = "method_body_size,lemma_body_size,annotations_per_method"

# Formal Disco seed (Claude opus + sonnet, pooled) and last Qwen iteration (it4).
_FD_CLAUDE_DAFNY = (f"{FINAL_AGENDAS}/agenda-dafny-sn-disco3__claude-opus__docs__10k.pkl"
                    f"+{FINAL_AGENDAS}/agenda-dafny-sn-disco3__claude-sonnet__docs__10k.pkl")
_FD_CLAUDE_VERUS = (f"{FINAL_AGENDAS}/agenda-verus-sn-disco3__claude-opus__docs__10k.pkl"
                    f"+{FINAL_AGENDAS}/agenda-verus-sn-disco3__claude-sonnet__docs__10k.pkl")


def _rarefaction_outputs(name: str) -> list[str]:
    return [f"plots/{name}-{grp}.{ext}"
            for grp in ("features1", "features2") for ext in ("svg", "png")]


def _run_rarefaction(name: str, sources: list[str]) -> None:
    """Run plot-rarefaction twice (features1, features2) over the same sources,
    reusing the on-disk feature cache so the second pass skips re-parsing."""
    for grp, metrics in (("features1", RAREFACTION_FEATURES1),
                         ("features2", RAREFACTION_FEATURES2)):
        _run("plot-rarefaction", *sources,
             "--metrics", metrics,
             "--repeats", "3",
             "--feature-cache", RAREFACTION_CACHE,
             "--name", f"{name}-{grp}")


# Comparison: Formal Disco (Claude seed + last Qwen iteration) vs. existing
# datasets. Dafny: DafnyBench. Verus: VerusBench, VeruSAGE, SAFE, VeruSyn.
RAREFACTION_COMPARISON_SOURCES = [
    # Dafny
    f"DafnyBench:{DAFNYBENCH_DIR}",
    f"Formal Disco - Claude:{_FD_CLAUDE_DAFNY}",
    f"Formal Disco - Qwen It=5:{FINAL_AGENDAS}/agenda-dafny-sn-disco3__qwen-docs-distilled_it4.pkl",
    # Verus
    (f"VerusBench:{VERUS_BENCH}/MBPP/verified+{VERUS_BENCH}/CloverBench/verified"
     f"+{VERUS_BENCH}/Diffy/verified+{VERUS_BENCH}/Misc/verified"),
    f"VeruSAGE:{VERUSAGE}/tasks",
    f"SAFE:{SAFE_DIR}",
    f"VeruSyn:{VERUSYN_DIR}",
    f"Formal Disco - Claude:{_FD_CLAUDE_VERUS}",
    f"Formal Disco - Qwen It=5:{FINAL_AGENDAS}/agenda-verus-sn-disco3__qwen-docs-distilled_it4.pkl",
]


@target(name="rarefaction-comparison",
        generates=_rarefaction_outputs("rarefaction-comparison"))
def _rarefaction_comparison() -> None:
    _run_rarefaction("rarefaction-comparison", RAREFACTION_COMPARISON_SOURCES)


# Self-improvement: all Formal Disco. Claude seed, then Qwen It=1..5 (it0..it4),
# for both languages, showing how diversity shifts across SFT iterations.
def _self_improvement_sources() -> list[str]:
    sources = [
        f"Claude Seed:{_FD_CLAUDE_DAFNY}",
        f"Claude Seed:{_FD_CLAUDE_VERUS}",
    ]
    for it in range(5):  # it0..it4  ->  It=1..It=5
        sources.append(
            f"Qwen It={it + 1}:{FINAL_AGENDAS}/agenda-dafny-sn-disco3__qwen-docs-distilled_it{it}.pkl")
        sources.append(
            f"Qwen It={it + 1}:{FINAL_AGENDAS}/agenda-verus-sn-disco3__qwen-docs-distilled_it{it}.pkl")
    return sources


@target(name="rarefaction-self-improvement",
        generates=_rarefaction_outputs("rarefaction-self-improvement"))
def _rarefaction_self_improvement() -> None:
    _run_rarefaction("rarefaction-self-improvement", _self_improvement_sources())


# ----- Ablations (paper Section: Ablations) -----------------------------------

ABLATION_AGENDAS = "agendas/for-ablations/final-runs"

# Ablation 2 (GitHub READMEs): theme diversity collapses without READMEs.
# Both runs are monolithic Claude Opus with 1k attempts each, so the curves
# are directly comparable; the log-x rarefaction handles the different number
# of verified programs (230 gh vs 128 nogh).
@target(name="ablation-gh-rarefaction",
        generates=[f"plots/ablation-gh-rarefaction.{ext}" for ext in ("svg", "png")])
def _ablation_gh_rarefaction() -> None:
    _run("plot-rarefaction",
         f"With READMEs:{ABLATION_AGENDAS}/agenda-dafny-sn-monolithic__gh__claude-opus.pkl",
         f"Without READMEs:{ABLATION_AGENDAS}/agenda-dafny-sn-monolithic__nogh__claude-opus.pkl",
         "--metrics", "subject_word",
         "--repeats", "3",
         "--feature-cache", RAREFACTION_CACHE,
         "--name", "ablation-gh-rarefaction")


# Ablation 3 (docs snippets): language-feature coverage collapses without
# documentation snippets in the seed context. Uses --statistic unique (species-
# accumulation curves) rather than entropy: the docs effect lives in the tail
# (features that never appear at all without docs), which bulk entropy hides.
# Dafny only (Opus + Sonnet pooled per condition, mirroring the 'Claude Seed'
# convention), so the figure pairs cleanly with the Dafny-only GH ablation.
# Verus numbers are quoted in the paper text instead; note that
# agenda-verus-sn-disco3__claude-sonnet__nodocs__10k.pkl actually contains
# Dafny programs (misnamed run), so Verus only has the Opus pair anyway.
@target(name="ablation-docs-rarefaction",
        generates=[f"plots/ablation-docs-rarefaction.{ext}" for ext in ("svg", "png")])
def _ablation_docs_rarefaction() -> None:
    _run("plot-rarefaction",
         f"With docs:{ABLATION_AGENDAS}/agenda-dafny-sn-disco3__claude-opus__docs__10k.pkl"
         f"+{ABLATION_AGENDAS}/agenda-dafny-sn-disco3__claude-sonnet__docs__10k.pkl",
         f"Without docs:{ABLATION_AGENDAS}/agenda-dafny-sn-disco3_claude-opus__nodocs__10k.pkl"
         f"+{ABLATION_AGENDAS}/agenda-dafny-sn-disco3_claude-sonnet__nodocs__10k.pkl",
         "--metrics", "language_features",
         "--statistic", "unique",
         "--repeats", "3",
         "--feature-cache", RAREFACTION_CACHE,
         "--name", "ablation-docs-rarefaction")


# ----- Fixer pass@k -----------------------------------------------------------

FIXER_DIR = "results/final"

# Claude (opus/sonnet) appear twice: a horizontal dashed line at pass@1 (for
# visual comparison against the models' pass@1) AND a solid pass@k curve (their
# evals ran to k=4, so plot-fixer-pass-at-k-faceted auto-clips the curve there).
# Qwen base + Disco/SAFE-SFT models -> pass@k curves with 95% CI bands.
# Dafny: 3 series (base Qwen, +Disco Qwen, Opus).
# Verus: 4 series (the above + SAFE-SFT Qwen).
@target(name="fixer-pass-at-k",
        generates=_outputs("fixer-pass-at-k"))
def _fixer_pass_at_k() -> None:
    _run(
        "plot-fixer-pass-at-k-faceted",
        "--horizontal",
        f"DafnyBench:Claude Opus 4.5:{FIXER_DIR}/dafny_fixer-eval-opus-4.5.json",
        f"VerusBench:Claude Opus 4.5:{FIXER_DIR}/verus_fixer-eval-opus-4.5.json",
        "--curve",
        f"DafnyBench:Qwen2.5-Coder-32B:{FIXER_DIR}/dafny_fixer-eval-qwen2.5-coder-32B.json",
        f"DafnyBench:Qwen2.5-Coder-32B + Disco SFT:{FIXER_DIR}/dafny_fixer-eval-qwen2.5-32B_fixer.json",
        f"VerusBench:Qwen2.5-Coder-32B:{FIXER_DIR}/verus_fixer-eval-qwen2.5-coder-32B.json",
        f"VerusBench:Qwen2.5-Coder-32B + Disco SFT:{FIXER_DIR}/verus_fixer-eval-qwen2.5-32B_verus_fixer.json",
        f"VerusBench:Qwen2.5-Coder-32B + SAFE SFT:{FIXER_DIR}/verus_fixer-eval-qwen2.5-32B_safe-fixer.json",
        # Claude curves (same labels as above -> same color; auto-clipped to k=4).
        f"DafnyBench:Claude Opus 4.5:{FIXER_DIR}/dafny_fixer-eval-opus-4.5.json",
        f"VerusBench:Claude Opus 4.5:{FIXER_DIR}/verus_fixer-eval-opus-4.5.json",
        "--ks", "1,2,4,8,16,32",
        "--name", "fixer-pass-at-k",
    )


# ----- Lemma proving pass@k ---------------------------------------------------

# Same design as the fixer plot, for the lemma-proving task on DafnyBench only.
# Faceted (single DafnyBench facet for now) so it stays consistent and gains the
# Verus facet later. Claude Opus: horizontal pass@1 line + curve (auto-clipped to
# k=4). Qwen base + Disco-SFT: pass@k curves to k=32.
@target(name="lemma-pass-at-k",
        generates=_outputs("lemma-pass-at-k"))
def _lemma_pass_at_k() -> None:
    _run(
        "plot-fixer-pass-at-k-faceted",
        "--horizontal",
        f"DafnyBench:Claude Opus 4.5:{FIXER_DIR}/dafny_lemma-eval-opus-4.5.json",
#        f"DafnyBench:Claude Sonnet 4.5:{FIXER_DIR}/dafny_lemma-eval-sonnet-4.5.json",
        "--curve",
        f"DafnyBench:Qwen2.5-Coder-32B:{FIXER_DIR}/dafny_lemma-eval-qwen2.5-coder-32B.json",
        f"DafnyBench:Qwen2.5-Coder-32B + Disco SFT:{FIXER_DIR}/dafny_lemma-eval-qwen2.5-32B_lemma_synth.json",
        # Claude curves (same labels as above -> same color; auto-clipped to k=4).
        f"DafnyBench:Claude Opus 4.5:{FIXER_DIR}/dafny_lemma-eval-opus-4.5.json",
#        f"DafnyBench:Claude Sonnet 4.5:{FIXER_DIR}/dafny_lemma-eval-sonnet-4.5.json",
        "--ks", "1,2,4,8,16,32",
        "--name", "lemma-pass-at-k",
        "--title", "Success rate at lemma proving task",
    )
