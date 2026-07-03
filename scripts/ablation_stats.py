"""Key numbers for the three system ablations (paper Section: Ablations).

1. Monolithic vs disco3 workers (Dafny, Claude Opus): verified-program yield
   per attempt + complexity of resulting programs.
2. GitHub READMEs vs none (monolithic, Claude Opus): yield + subject-word
   (theme) diversity at matched sample size, top themes to show collapse.
3. Docs snippets vs none (disco3, Dafny+Verus, Opus+Sonnet): yield + language
   feature coverage at matched sample size, features that never appear.

Run from the repo root:
    venv/bin/python scripts/ablation_stats.py
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import load_agenda, _extract_verified_texts, _extract_features_parallel
from language import Language

ABLATIONS_DIR = Path("agendas/for-ablations/final-runs")

RUNS = {
    "disco3-dafny-opus": ("agenda-dafny-sn-disco3__claude-opus__docs__10k.pkl", Language.DAFNY),
    "mono-gh-dafny-opus": ("agenda-dafny-sn-monolithic__gh__claude-opus.pkl", Language.DAFNY),
    "mono-nogh-dafny-opus": ("agenda-dafny-sn-monolithic__nogh__claude-opus.pkl", Language.DAFNY),
    "disco3-dafny-opus-nodocs": ("agenda-dafny-sn-disco3_claude-opus__nodocs__10k.pkl", Language.DAFNY),
    "disco3-dafny-sonnet": ("agenda-dafny-sn-disco3__claude-sonnet__docs__10k.pkl", Language.DAFNY),
    "disco3-dafny-sonnet-nodocs": ("agenda-dafny-sn-disco3_claude-sonnet__nodocs__10k.pkl", Language.DAFNY),
    "disco3-verus-opus": ("agenda-verus-sn-disco3__claude-opus__docs__10k.pkl", Language.VERUS),
    "disco3-verus-opus-nodocs": ("agenda-verus-sn-disco3__claude-opus__nodocs__10k.pkl", Language.VERUS),
    "disco3-verus-sonnet": ("agenda-verus-sn-disco3__claude-sonnet__docs__10k.pkl", Language.VERUS),
    # NOTE: agenda-verus-sn-disco3__claude-sonnet__nodocs__10k.pkl is excluded:
    # despite its name it contains Dafny programs (objects typed 'dafny-program',
    # program text says "this Dafny program..."), so the Verus docs ablation
    # only has the Opus pair.
}

SUBSAMPLE_REPEATS = 20
SEED = 0


def load_run(name: str):
    fname, lang = RUNS[name]
    agenda = load_agenda(str(ABLATIONS_DIR / fname))
    texts = _extract_verified_texts(agenda)
    features = _extract_features_parallel(texts, lang, desc=name)
    return agenda, lang, texts, features


def attempts_and_yield(agenda: dict) -> tuple[int, int]:
    attempts = sum(sum(oc.values()) for oc in agenda["task_outcomes"].values())
    n_verified = sum(1 for k in agenda["objects"] if k.startswith("dataset/"))
    return attempts, n_verified


def pooled(features: list[dict[str, Counter]], metric: str) -> Counter:
    c = Counter()
    for ft in features:
        c.update(ft[metric])
    return c


def entropy(counter: Counter) -> float:
    import math
    total = sum(counter.values())
    return -sum((v / total) * math.log2(v / total) for v in counter.values())


def matched_subsample_stats(
    features: list[dict[str, Counter]], metric: str, n: int,
) -> tuple[float, float]:
    """Mean (entropy, unique values) of `metric` over SUBSAMPLE_REPEATS
    subsamples of size n."""
    rng = random.Random(SEED)
    ents, uniqs = [], []
    for _ in range(SUBSAMPLE_REPEATS):
        sample = rng.sample(features, n)
        c = pooled(sample, metric)
        ents.append(entropy(c))
        uniqs.append(len(c))
    return sum(ents) / len(ents), sum(uniqs) / len(uniqs)


def loc(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    data = {name: load_run(name) for name in RUNS}

    # ---- Ablation 1: monolithic vs disco3 -----------------------------------
    header("ABLATION 1: monolithic agent vs disco3 workers (Dafny, Claude Opus, with GH+docs)")
    for name in ["mono-gh-dafny-opus", "disco3-dafny-opus"]:
        agenda, _, texts, _ = data[name]
        attempts, n_verified = attempts_and_yield(agenda)
        locs = sorted(loc(t) for t in texts)
        print(f"{name:28s} attempts={attempts:6d}  verified={n_verified:5d}  "
              f"yield/attempt={pct(n_verified / attempts)}")
        print(f"{'':28s} program LOC: mean={sum(locs)/len(locs):6.1f}  "
              f"median={locs[len(locs)//2]:4d}  p90={locs[int(0.9*len(locs))]:4d}  "
              f"max={locs[-1]:5d}  (n={len(locs)} longest-per-idea)")

    # ---- Ablation 2: GitHub READMEs vs none ---------------------------------
    header("ABLATION 2: GitHub READMEs vs no READMEs (monolithic, Dafny, Claude Opus)")
    n_match = min(len(data[n][3]) for n in ["mono-gh-dafny-opus", "mono-nogh-dafny-opus"])
    print(f"(diversity compared at matched sample size N={n_match}, "
          f"{SUBSAMPLE_REPEATS} subsamples)\n")
    for name in ["mono-gh-dafny-opus", "mono-nogh-dafny-opus"]:
        agenda, _, _, features = data[name]
        attempts, n_verified = attempts_and_yield(agenda)
        ent, uniq = matched_subsample_stats(features, "subject_word", n_match)
        print(f"{name:28s} verified={n_verified:4d}/{attempts} ({pct(n_verified/attempts)})  "
              f"subject-word entropy@{n_match}={ent:.2f} bits  unique@{n_match}={uniq:.0f}")
        top = pooled(features, "subject_word").most_common(10)
        total = sum(pooled(features, "subject_word").values())
        top_str = ", ".join(f"{w} ({pct(c/total)})" for w, c in top)
        print(f"{'':28s} top-10 subject words: {top_str}")

    # ---- Ablation 3: docs snippets vs none ----------------------------------
    header("ABLATION 3: documentation snippets vs none (disco3, Dafny+Verus, Opus+Sonnet)")
    pairs = [
        ("disco3-dafny-opus", "disco3-dafny-opus-nodocs"),
        ("disco3-dafny-sonnet", "disco3-dafny-sonnet-nodocs"),
        ("disco3-verus-opus", "disco3-verus-opus-nodocs"),
    ]
    for docs_name, nodocs_name in pairs:
        n_match = min(len(data[docs_name][3]), len(data[nodocs_name][3]))
        print(f"\n--- {docs_name} vs {nodocs_name} (matched N={n_match}) ---")
        feats_used = {}
        for name in (docs_name, nodocs_name):
            agenda, lang, _, features = data[name]
            attempts, n_verified = attempts_and_yield(agenda)
            ent, uniq = matched_subsample_stats(features, "language_features", n_match)
            used = pooled(features, "language_features")
            feats_used[name] = used
            n_known = len(lang.get_backend()._language_feature_regexes)
            print(f"{name:30s} verified={n_verified:5d}/{attempts} ({pct(n_verified/attempts)})  "
                  f"lang-feature entropy@{n_match}={ent:.2f} bits  "
                  f"features used: {len(used)}/{n_known}")
        docs_only = set(feats_used[docs_name]) - set(feats_used[nodocs_name])
        nodocs_only = set(feats_used[nodocs_name]) - set(feats_used[docs_name])
        print(f"  features appearing ONLY with docs ({len(docs_only)}): "
              f"{', '.join(sorted(docs_only)[:15])}{' ...' if len(docs_only) > 15 else ''}")
        print(f"  features appearing ONLY without docs ({len(nodocs_only)}): "
              f"{', '.join(sorted(nodocs_only))}")


if __name__ == "__main__":
    main()
