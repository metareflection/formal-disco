"""Tests for the FramaCBackend (language/framac/) and FramaCPromptBuilder."""

import json
import random
from pathlib import Path
from collections import Counter

from language import Language, Program, VerificationOutcome
from language.framac import FramaCBackend
from language.framac.prompt import FramaCPromptBuilder

EXAMPLES = Path(__file__).parent.parent / "language" / "framac" / "examples"
FEATURES = Path(__file__).parent.parent / "language" / "framac" / "features"


def load(name: str) -> Program:
    source = (EXAMPLES / name).read_text()
    return Program(source, Language.FRAMAC, name=name)


def backend() -> FramaCBackend:
    return FramaCBackend()


class TestVerify:
    """End-to-end verification tests — require frama-c (with WP) to be installed."""

    def test_max_succeeds(self):
        result = load("max.c").verify()
        assert result.outcome == VerificationOutcome.SUCCESS
        assert result.status == 0

    def test_curated_swap_succeeds(self):
        result = load("curated/swap.c").verify()
        assert result.outcome == VerificationOutcome.SUCCESS

    def test_invalid_goal_unproven(self):
        # An unprovable goal maps to GOAL_UNPROVEN, not FAIL.
        result = load("invalid.c").verify()
        assert result.outcome == VerificationOutcome.GOAL_UNPROVEN

    def test_syntax_error_fails(self):
        result = Program("int broken( { }", Language.FRAMAC).verify()
        assert result.outcome == VerificationOutcome.FAIL

    def test_verify_returns_output(self):
        result = load("max.c").verify()
        assert result.stdout or result.stderr

    def test_verify_inline_program(self):
        source = (
            "/*@ requires \\true;\n"
            "    assigns \\nothing;\n"
            "    ensures \\result == x + 1; */\n"
            "int inc(int x) { return x + 1; }\n"
        )
        result = Program(source, Language.FRAMAC).verify()
        assert result.outcome == VerificationOutcome.SUCCESS


class TestFeatureSets:
    """Feature-set extraction tests — pure Python, no frama-c invocation."""

    def test_returns_all_keys(self):
        b = backend()
        assert set(b.feature_sets(load("max.c")).keys()) == b.feature_metrics

    def test_all_values_are_counters(self):
        fs = backend().feature_sets(load("curated/copy.c"))
        assert all(isinstance(v, Counter) for v in fs.values())

    def test_subject_words_from_function_names(self):
        # sum_array.c defines a function whose name contributes subject words.
        words = backend().feature_sets(load("sum_array.c"))["subject_word"]
        assert isinstance(words, Counter)
        assert sum(words.values()) > 0

    def test_annotation_templates_namespaced(self):
        templates = backend().feature_sets(load("max.c"))["annotation_template"]
        assert len(templates) > 0
        # Templates are namespaced by clause kind.
        assert all(":" in t for t in templates)
        assert any(t.startswith(("requires:", "ensures:", "assert:", "invariant:"))
                   for t in templates)

    def test_method_body_size_counts_functions(self):
        # Every curated example defines at least one C function.
        sizes = backend().feature_sets(load("curated/clamp.c"))["method_body_size"]
        assert sum(sizes.values()) >= 1

    def test_lemma_body_size_counts_logic_decls(self):
        # clamp.c carries six ACSL lemmas.
        sizes = backend().feature_sets(load("curated/clamp.c"))["lemma_body_size"]
        assert sum(sizes.values()) == 6

    def test_loop_skeleton_present_for_loops(self):
        skeletons = backend().feature_sets(load("curated/copy.c"))["loop_skeleton"]
        assert sum(skeletons.values()) >= 1

    def test_loop_skeleton_absent_without_loops(self):
        skeletons = backend().feature_sets(load("max.c"))["loop_skeleton"]
        assert len(skeletons) == 0

    def test_language_features_detected(self):
        lf = backend().feature_sets(load("curated/copy.c"))["language_features"]
        assert lf["requires"] >= 1
        assert lf["ensures"] >= 1
        assert lf["loop-invariant"] >= 1

    def test_empty_program(self):
        fs = backend().feature_sets(Program("", Language.FRAMAC))
        assert set(fs.keys()) == backend().feature_metrics
        assert all(len(v) == 0 for v in fs.values())

    def test_feature_sets_invariant_to_strip(self):
        b = backend()
        for name in ("max.c", "sum_array.c", "binary_search.c",
                     "curated/clamp.c", "curated/copy.c", "curated/iota.c"):
            prog = load(name)
            assert b.feature_sets(prog) == b.feature_sets(b.strip(prog)), name


class TestFeatureMetrics:
    def test_feature_metrics_nonempty(self):
        assert len(backend().feature_metrics) > 0

    def test_no_legacy_complexity_api(self):
        # complexity()/complexity_metrics were removed from the interface.
        b = backend()
        assert not hasattr(b, "complexity")
        assert not hasattr(b, "complexity_metrics")


class TestDocSnippets:
    def test_features_dir_exists(self):
        assert backend().features_dir == FEATURES
        assert FEATURES.is_dir()

    def test_doc_snippets_match_language_features(self):
        b = backend()
        ids = {e["id"] for e in json.load(
            (FEATURES / "language-features.json").open())}
        assert set(b.doc_snippets.keys()) == ids

    def test_doc_snippets_nonempty(self):
        for fid, text in backend().doc_snippets.items():
            assert text.strip(), fid

    def test_sample_doc_snippets_count_and_distinct(self):
        b = backend()
        chosen = b.sample_doc_snippets(random.Random(0), 5)
        assert len(chosen) == 5
        assert len({fid for fid, _ in chosen}) == 5

    def test_sample_doc_snippets_clamped_to_available(self):
        b = backend()
        chosen = b.sample_doc_snippets(random.Random(0), 10_000)
        assert len(chosen) == len(b.doc_snippets)


class TestPromptBuilder:
    def test_implement(self):
        msgs = FramaCPromptBuilder().implement(idea="Clamp a value to a range.")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "Clamp a value" in msgs[1]["content"]

    def test_initiate_accepts_doc_snippets(self):
        # Regression: main passes doc_snippets= into initiate(); it must accept it.
        pb = FramaCPromptBuilder()
        snippets = [("loop-invariant", "loop invariant 0 <= i;")]
        msgs = pb.initiate(repo="acme/widgets", readme="A widget toolkit.",
                           doc_snippets=snippets)
        assert "acme/widgets" in msgs[1]["content"]
        assert "loop-invariant" in msgs[1]["content"]

    def test_initiate_without_doc_snippets(self):
        msgs = FramaCPromptBuilder().initiate(repo="r", readme="readme")
        assert msgs[1]["role"] == "user"
