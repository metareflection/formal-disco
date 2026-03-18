"""Tests for the DafnyBackend (language/dafny/) and DafnyPromptBuilder."""

from pathlib import Path
from collections import Counter

from language import Language, Program, VerificationOutcome
from language.dafny import DafnyBackend
from language.dafny.prompt import DafnyPromptBuilder

EXAMPLES = Path(__file__).parent.parent / "language" / "dafny" / "examples"


def load(name: str) -> Program:
    source = (EXAMPLES / name).read_text()
    return Program(source, Language.DAFNY, name=name)


def backend() -> DafnyBackend:
    return DafnyBackend()


class TestVerify:
    """End-to-end verification tests — require Dafny to be installed."""

    def test_max_succeeds(self):
        result = load("max.dfy").verify()
        assert result.outcome == VerificationOutcome.SUCCESS
        assert result.status == 0

    def test_binary_search_succeeds(self):
        result = load("binary_search.dfy").verify()
        assert result.outcome == VerificationOutcome.SUCCESS

    def test_sum_array_succeeds(self):
        result = load("sum_array.dfy").verify()
        assert result.outcome == VerificationOutcome.SUCCESS

    def test_fibonacci_succeeds(self):
        result = load("fibonacci.dfy").verify()
        assert result.outcome == VerificationOutcome.SUCCESS

    def test_invalid_goal_unproven(self):
        # An unprovable postcondition maps to GOAL_UNPROVEN, not FAIL.
        result = load("invalid.dfy").verify()
        assert result.outcome == VerificationOutcome.GOAL_UNPROVEN

    def test_syntax_error_fails(self):
        prog = Program("method Broken( { }", Language.DAFNY)
        result = prog.verify()
        assert result.outcome == VerificationOutcome.FAIL

    def test_verify_returns_stdout(self):
        result = load("max.dfy").verify()
        assert result.stdout

    def test_verify_inline_program(self):
        source = "method Double(x: int) returns (r: int)\n  ensures r == 2 * x\n{ r := 2 * x; }\n"
        result = Program(source, Language.DAFNY).verify()
        assert result.outcome == VerificationOutcome.SUCCESS


class TestFeatureSets:
    """Feature-set extraction tests — pure Python, no Dafny invocation."""

    def test_returns_all_keys(self):
        b = backend()
        assert set(b.feature_sets(load("max.dfy")).keys()) == b.feature_metrics

    def test_subject_words_are_counters(self):
        assert isinstance(backend().feature_sets(load("max.dfy"))["subject_words"], Counter)

    def test_subject_words_from_decl_names(self):
        words = backend().feature_sets(load("binary_search.dfy"))["subject_words"]
        # "BinarySearch" → split → ['binary', 'search']
        # 'binary' is an adjective → filtered by POS tagger; 'search' is a noun → kept
        assert "binary" not in words
        assert "search" in words

    def test_invariant_templates_extracted(self):
        invs = backend().feature_sets(load("binary_search.dfy"))["invariant_templates"]
        assert len(invs) > 0
        for template in invs:
            assert isinstance(template, str)

    def test_ensures_templates_extracted(self):
        assert len(backend().feature_sets(load("max.dfy"))["ensures_templates"]) > 0

    def test_requires_templates_extracted(self):
        assert len(backend().feature_sets(load("binary_search.dfy"))["requires_templates"]) > 0

    def test_loop_skeletons_for_loops(self):
        skeletons = backend().feature_sets(load("binary_search.dfy"))["loop_skeletons"]
        assert len(skeletons) > 0
        assert any("while" in sk for sk in skeletons)

    def test_loop_skeletons_empty_for_no_loops(self):
        assert len(backend().feature_sets(load("max.dfy"))["loop_skeletons"]) == 0

    def test_nested_loops_skeleton(self):
        source = (
            "method DoubleLoop(n: int) {\n"
            "  var i := 0;\n"
            "  while i < n {\n"
            "    var j := 0;\n"
            "    while j < n {\n"
            "      j := j + 1;\n"
            "    }\n"
            "    i := i + 1;\n"
            "  }\n"
            "}\n"
        )
        skeletons = backend().feature_sets(Program(source, Language.DAFNY))["loop_skeletons"]
        assert any("while" in sk and sk.count("while") == 2 for sk in skeletons)

    def test_template_replaces_identifiers(self):
        source = (
            "method Foo(x: int) returns (y: int)\n"
            "  ensures y == x + 1\n"
            "{ y := x + 1; }\n"
        )
        templates = backend().feature_sets(Program(source, Language.DAFNY))["ensures_templates"]
        for tmpl in templates:
            assert "x" not in tmpl
            assert "y" not in tmpl
            assert "*" in tmpl

    def test_empty_program(self):
        fs = backend().feature_sets(Program("", Language.DAFNY))
        for v in fs.values():
            assert len(v) == 0


class TestComplexity:
    """Complexity metric tests — pure Python, no Dafny invocation."""

    def test_returns_all_keys(self):
        b = backend()
        assert set(b.complexity(load("max.dfy")).keys()) == b.complexity_metrics

    def test_body_sizes_positive(self):
        c = backend().complexity(load("binary_search.dfy"))
        assert all(s > 0 for s in c["body_sizes"])

    def test_no_loops_in_max(self):
        c = backend().complexity(load("max.dfy"))
        assert all(n == 0 for n in c["n_loops_per_method"])

    def test_loops_in_binary_search(self):
        c = backend().complexity(load("binary_search.dfy"))
        assert any(n > 0 for n in c["n_loops_per_method"])

    def test_idents_in_asserts(self):
        source = "method Check(x: int) {\n  assert x > 0;\n}\n"
        c = backend().complexity(Program(source, Language.DAFNY))
        assert len(c["n_idents_in_asserts"]) == 1
        assert c["n_idents_in_asserts"][0] >= 1

    def test_multiple_functions_body_sizes(self):
        c = backend().complexity(load("sum_array.dfy"))
        assert len(c["body_sizes"]) >= 2


class TestPromptBuilder:
    """Tests for DafnyPromptBuilder via backend.prompt_builder."""

    def test_backend_exposes_prompt_builder(self):
        pb = backend().prompt_builder
        assert isinstance(pb, DafnyPromptBuilder)

    def test_implement_returns_two_messages(self):
        msgs = backend().prompt_builder.implement(idea="Sort an array in ascending order.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_implement_contains_idea(self):
        idea = "Compute the GCD of two natural numbers."
        msgs = backend().prompt_builder.implement(idea=idea)
        assert idea in msgs[1]["content"]

    def test_implement_system_mentions_dafny(self):
        msgs = backend().prompt_builder.implement(idea="anything")
        assert "Dafny" in msgs[0]["content"]

    def test_repair_returns_two_messages(self):
        msgs = backend().prompt_builder.repair(
            program="method Foo() {}",
            notes="postcondition could not be proved",
            example_before="before",
            example_diff="@@ ...",
            example_after="after",
        )
        assert len(msgs) == 2

    def test_repair_embeds_example(self):
        msgs = backend().prompt_builder.repair(
            program="method Foo() {}",
            notes="error",
            example_before="BEFORE_TEXT",
            example_diff="DIFF_TEXT",
            example_after="AFTER_TEXT",
        )
        assert "BEFORE_TEXT" in msgs[0]["content"]
        assert "DIFF_TEXT" in msgs[0]["content"]
        assert "AFTER_TEXT" in msgs[0]["content"]

    def test_repair_user_contains_program_and_notes(self):
        prog = "method Bar() {}"
        notes = "assertion might not hold"
        msgs = backend().prompt_builder.repair(
            program=prog, notes=notes,
            example_before="b", example_diff="d", example_after="a",
        )
        assert prog in msgs[1]["content"]
        assert notes in msgs[1]["content"]

    def test_extend_returns_two_messages(self):
        msgs = backend().prompt_builder.extend(
            program="method Foo() {}",
            example_before="b", example_diff="d", example_after="a",
        )
        assert len(msgs) == 2

    def test_extend_user_contains_program(self):
        prog = "method Baz() {}"
        msgs = backend().prompt_builder.extend(
            program=prog,
            example_before="b", example_diff="d", example_after="a",
        )
        assert prog in msgs[1]["content"]

    def test_idea_returns_two_messages(self):
        msgs = backend().prompt_builder.idea(repo="mylib", readme="A sorting library.")
        assert len(msgs) == 2

    def test_idea_user_contains_repo_and_readme(self):
        msgs = backend().prompt_builder.idea(repo="REPO_NAME", readme="README_CONTENT")
        assert "REPO_NAME" in msgs[1]["content"]
        assert "README_CONTENT" in msgs[1]["content"]

    def test_language_enum_prompt_builder(self):
        pb = Language.DAFNY.get_backend().prompt_builder
        assert isinstance(pb, DafnyPromptBuilder)


class TestStrip:
    """Tests for LanguageBackend.strip() and invariance of metrics to stripping."""

    def test_strip_removes_line_comments(self):
        source = "method Foo() {\n  var x := 0; // set x to zero\n}\n"
        stripped = backend().strip(Program(source, Language.DAFNY))
        assert "//" not in str(stripped)

    def test_strip_removes_block_comments(self):
        source = "method Foo() {\n  /* do nothing */\n  var x := 0;\n}\n"
        stripped = backend().strip(Program(source, Language.DAFNY))
        assert "/*" not in str(stripped)
        assert "*/" not in str(stripped)

    def test_strip_removes_blank_lines(self):
        source = "method Foo() {\n\n  var x := 0;\n\n}\n"
        lines = str(backend().strip(Program(source, Language.DAFNY))).split('\n')
        assert all(ln.strip() for ln in lines)

    def test_strip_preserves_language(self):
        source = "method Foo() {}\n"
        prog = Program(source, Language.DAFNY)
        stripped = backend().strip(prog)
        assert stripped.language == Language.DAFNY

    def test_strip_preserves_name(self):
        source = "method Foo() {}\n"
        prog = Program(source, Language.DAFNY, name="test.dfy")
        stripped = backend().strip(prog)
        assert stripped.name == "test.dfy"

    def test_assert_in_line_comment_not_counted(self):
        b = backend()
        source = (
            "method Foo(x: int) {\n"
            "  // assert x > 0;\n"
            "  var y := x;\n"
            "}\n"
        )
        c = b.complexity(Program(source, Language.DAFNY))
        assert c["n_idents_in_asserts"] == []

    def test_invariant_in_line_comment_not_counted(self):
        b = backend()
        source = (
            "method Foo(n: int) {\n"
            "  var i := 0;\n"
            "  while i < n\n"
            "    // invariant i >= 0\n"
            "    decreases n - i\n"
            "  {\n"
            "    i := i + 1;\n"
            "  }\n"
            "}\n"
        )
        fs = b.feature_sets(Program(source, Language.DAFNY))
        assert len(fs["invariant_templates"]) == 0

    def test_assert_in_block_comment_not_counted(self):
        b = backend()
        source = (
            "method Foo(x: int) {\n"
            "  /* assert x > 0; */\n"
            "  var y := x;\n"
            "}\n"
        )
        c = b.complexity(Program(source, Language.DAFNY))
        assert c["n_idents_in_asserts"] == []

    def test_ensures_in_comment_not_counted(self):
        b = backend()
        source = (
            "// ensures result >= 0\n"
            "method Foo() returns (result: int) {\n"
            "  result := 1;\n"
            "}\n"
        )
        fs = b.feature_sets(Program(source, Language.DAFNY))
        assert len(fs["ensures_templates"]) == 0

    def test_complexity_invariant_to_strip(self):
        b = backend()
        source = (
            "method Sum(n: int) returns (s: int)\n"
            "  requires n >= 0\n"
            "  ensures s >= 0\n"
            "{\n"
            "  s := 0;\n"
            "  var i := 0;\n"
            "  while i < n\n"
            "    invariant 0 <= i <= n  // i stays in range\n"
            "    invariant s >= 0\n"
            "    decreases n - i\n"
            "  {\n"
            "    assert s >= 0;  /* holds by invariant */\n"
            "    s := s + i;\n"
            "    i := i + 1;\n"
            "  }\n"
            "}\n"
        )
        prog = Program(source, Language.DAFNY)
        stripped = b.strip(prog)
        assert b.complexity(prog) == b.complexity(stripped)

    def test_feature_sets_invariant_to_strip(self):
        b = backend()
        source = (
            "method BinarySearch(a: array<int>, target: int) returns (idx: int)\n"
            "  requires forall i, j :: 0 <= i < j < a.Length ==> a[i] <= a[j]\n"
            "  ensures idx >= 0 ==> a[idx] == target  // found\n"
            "{\n"
            "  /* simple linear scan for now */\n"
            "  idx := -1;\n"
            "  var i := 0;\n"
            "  while i < a.Length\n"
            "    invariant 0 <= i  // lower bound\n"
            "    invariant idx == -1\n"
            "  {\n"
            "    if a[i] == target { idx := i; return; }\n"
            "    i := i + 1;\n"
            "  }\n"
            "}\n"
        )
        prog = Program(source, Language.DAFNY)
        stripped = b.strip(prog)
        assert b.feature_sets(prog) == b.feature_sets(stripped)


class TestBackendMetadata:
    """Backend registration and metric-set membership."""

    def test_complexity_metrics_nonempty(self):
        assert len(backend().complexity_metrics) > 0

    def test_feature_metrics_nonempty(self):
        assert len(backend().feature_metrics) > 0

    def test_language_enum_returns_dafny_backend(self):
        assert type(Language.DAFNY.get_backend()).__name__ == "DafnyBackend"
