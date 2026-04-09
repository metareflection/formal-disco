"""Tests for the VerusBackend (language/verus/) and VerusPromptBuilder."""

from pathlib import Path
from collections import Counter

from language import Language, Program, VerificationOutcome
from language.verus import VerusBackend
from language.verus.prompt import VerusPromptBuilder

EXAMPLES = Path(__file__).parent.parent / "language" / "verus" / "examples"


def load(name: str) -> Program:
    source = (EXAMPLES / name).read_text()
    return Program(source, Language.VERUS, name=name)


def backend() -> VerusBackend:
    return VerusBackend()


class TestVerify:
    """End-to-end verification tests — require Verus to be installed."""

    def test_max_succeeds(self):
        result = load("max.rs").verify()
        assert result.outcome == VerificationOutcome.SUCCESS
        assert result.status == 0

    def test_binary_search_succeeds(self):
        result = load("binary_search.rs").verify()
        assert result.outcome == VerificationOutcome.SUCCESS

    def test_sum_vec_succeeds(self):
        result = load("sum_vec.rs").verify()
        assert result.outcome == VerificationOutcome.SUCCESS

    def test_invalid_goal_unproven(self):
        result = load("invalid.rs").verify()
        assert result.outcome in (VerificationOutcome.GOAL_UNPROVEN, VerificationOutcome.FAIL)

    def test_syntax_error_fails(self):
        prog = Program("fn broken( { }", Language.VERUS)
        result = prog.verify()
        assert result.outcome == VerificationOutcome.FAIL

    def test_verify_returns_stdout(self):
        result = load("max.rs").verify()
        assert result.stdout or result.stderr


class TestFeatureSets:
    """Feature-set extraction tests — pure Python, no Verus invocation."""

    def test_returns_all_keys(self):
        b = backend()
        assert set(b.feature_sets(load("max.rs")).keys()) == b.feature_metrics

    def test_subject_words_are_counters(self):
        assert isinstance(backend().feature_sets(load("max.rs"))["subject_words"], Counter)

    def test_invariant_templates_extracted(self):
        invs = backend().feature_sets(load("binary_search.rs"))["invariant_templates"]
        assert len(invs) > 0
        for template in invs:
            assert isinstance(template, str)

    def test_ensures_templates_extracted(self):
        assert len(backend().feature_sets(load("max.rs"))["ensures_templates"]) > 0

    def test_requires_templates_extracted(self):
        assert len(backend().feature_sets(load("binary_search.rs"))["requires_templates"]) > 0

    def test_loop_skeletons_for_loops(self):
        skeletons = backend().feature_sets(load("binary_search.rs"))["loop_skeletons"]
        assert len(skeletons) > 0
        assert any("while" in sk for sk in skeletons)

    def test_loop_skeletons_empty_for_no_loops(self):
        assert len(backend().feature_sets(load("max.rs"))["loop_skeletons"]) == 0

    def test_nested_loops_skeleton(self):
        source = (
            "verus! {\n"
            "fn double_loop(n: u64) {\n"
            "  let mut i: u64 = 0;\n"
            "  while i < n\n"
            "  {\n"
            "    let mut j: u64 = 0;\n"
            "    while j < n\n"
            "    {\n"
            "      j = j + 1;\n"
            "    }\n"
            "    i = i + 1;\n"
            "  }\n"
            "}\n"
            "} // verus!\n"
        )
        skeletons = backend().feature_sets(Program(source, Language.VERUS))["loop_skeletons"]
        assert any("while" in sk and sk.count("while") == 2 for sk in skeletons)

    def test_template_replaces_identifiers(self):
        source = (
            "verus! {\n"
            "fn foo(x: i64) -> (y: i64)\n"
            "  ensures y == x + 1\n"
            "{ x + 1 }\n"
            "} // verus!\n"
        )
        templates = backend().feature_sets(Program(source, Language.VERUS))["ensures_templates"]
        for tmpl in templates:
            assert "x" not in tmpl
            assert "y" not in tmpl
            assert "*" in tmpl

    def test_empty_program(self):
        fs = backend().feature_sets(Program("", Language.VERUS))
        for v in fs.values():
            assert len(v) == 0


class TestComplexity:
    """Complexity metric tests — pure Python, no Verus invocation."""

    def test_returns_all_keys(self):
        b = backend()
        assert set(b.complexity(load("max.rs")).keys()) == b.complexity_metrics

    def test_body_sizes_positive(self):
        c = backend().complexity(load("binary_search.rs"))
        assert all(s > 0 for s in c["body_sizes"])

    def test_no_loops_in_max(self):
        c = backend().complexity(load("max.rs"))
        assert all(n == 0 for n in c["n_loops_per_fn"])

    def test_loops_in_binary_search(self):
        c = backend().complexity(load("binary_search.rs"))
        assert any(n > 0 for n in c["n_loops_per_fn"])

    def test_idents_in_asserts(self):
        source = "verus! {\nfn check(x: i64) {\n  assert(x > 0);\n}\n}\n"
        c = backend().complexity(Program(source, Language.VERUS))
        assert len(c["n_idents_in_asserts"]) == 1
        assert c["n_idents_in_asserts"][0] >= 1

    def test_multiple_fns_body_sizes(self):
        c = backend().complexity(load("sum_vec.rs"))
        assert len(c["body_sizes"]) >= 2


class TestPromptBuilder:
    """Tests for VerusPromptBuilder via backend.prompt_builder."""

    def test_backend_exposes_prompt_builder(self):
        pb = backend().prompt_builder
        assert isinstance(pb, VerusPromptBuilder)

    def test_implement_returns_two_messages(self):
        msgs = backend().prompt_builder.implement(idea="Sort a vector in ascending order.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_implement_contains_idea(self):
        idea = "Compute the GCD of two natural numbers."
        msgs = backend().prompt_builder.implement(idea=idea)
        assert idea in msgs[1]["content"]

    def test_implement_system_mentions_verus(self):
        msgs = backend().prompt_builder.implement(idea="anything")
        assert "Verus" in msgs[0]["content"]

    def test_repair_returns_two_messages(self):
        msgs = backend().prompt_builder.repair(
            program="fn foo() {}",
            notes="postcondition could not be proved",
            example_before="before",
            example_diff="@@ ...",
            example_after="after",
        )
        assert len(msgs) == 2

    def test_repair_embeds_example(self):
        msgs = backend().prompt_builder.repair(
            program="fn foo() {}",
            notes="error",
            example_before="BEFORE_TEXT",
            example_diff="DIFF_TEXT",
            example_after="AFTER_TEXT",
        )
        assert "BEFORE_TEXT" in msgs[0]["content"]
        assert "DIFF_TEXT" in msgs[0]["content"]
        assert "AFTER_TEXT" in msgs[0]["content"]

    def test_repair_user_contains_program_and_notes(self):
        prog = "fn bar() {}"
        notes = "assertion might not hold"
        msgs = backend().prompt_builder.repair(
            program=prog, notes=notes,
            example_before="b", example_diff="d", example_after="a",
        )
        assert prog in msgs[1]["content"]
        assert notes in msgs[1]["content"]

    def test_extend_returns_two_messages(self):
        msgs = backend().prompt_builder.extend(
            program="fn foo() {}",
            example_before="b", example_diff="d", example_after="a",
        )
        assert len(msgs) == 2

    def test_extend_user_contains_program(self):
        prog = "fn baz() {}"
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
        pb = Language.VERUS.get_backend().prompt_builder
        assert isinstance(pb, VerusPromptBuilder)


class TestStrip:
    """Tests for LanguageBackend.strip() and invariance of metrics to stripping."""

    def test_strip_removes_line_comments(self):
        source = "fn foo() {\n  let x: i64 = 0; // set x to zero\n}\n"
        stripped = backend().strip(Program(source, Language.VERUS))
        assert "//" not in str(stripped)

    def test_strip_removes_block_comments(self):
        source = "fn foo() {\n  /* do nothing */\n  let x: i64 = 0;\n}\n"
        stripped = backend().strip(Program(source, Language.VERUS))
        assert "/*" not in str(stripped)
        assert "*/" not in str(stripped)

    def test_strip_removes_blank_lines(self):
        source = "fn foo() {\n\n  let x: i64 = 0;\n\n}\n"
        lines = str(backend().strip(Program(source, Language.VERUS))).split('\n')
        assert all(ln.strip() for ln in lines)

    def test_strip_preserves_language(self):
        source = "fn foo() {}\n"
        prog = Program(source, Language.VERUS)
        stripped = backend().strip(prog)
        assert stripped.language == Language.VERUS

    def test_strip_preserves_name(self):
        source = "fn foo() {}\n"
        prog = Program(source, Language.VERUS, name="test.rs")
        stripped = backend().strip(prog)
        assert stripped.name == "test.rs"

    def test_assert_in_line_comment_not_counted(self):
        b = backend()
        source = (
            "fn foo(x: i64) {\n"
            "  // assert(x > 0);\n"
            "  let y = x;\n"
            "}\n"
        )
        c = b.complexity(Program(source, Language.VERUS))
        assert c["n_idents_in_asserts"] == []

    def test_ensures_in_comment_not_counted(self):
        b = backend()
        source = (
            "// ensures result >= 0\n"
            "fn foo() -> (result: i64) {\n"
            "  1\n"
            "}\n"
        )
        fs = b.feature_sets(Program(source, Language.VERUS))
        assert len(fs["ensures_templates"]) == 0

    def test_complexity_invariant_to_strip(self):
        b = backend()
        source = (
            "verus! {\n"
            "fn sum(n: u64) -> (s: u64)\n"
            "  requires n < 1000\n"
            "  ensures s >= 0\n"
            "{\n"
            "  let mut s: u64 = 0;\n"
            "  let mut i: u64 = 0;\n"
            "  while i < n\n"
            "    invariant 0 <= i <= n  // i stays in range\n"
            "    invariant s >= 0\n"
            "    decreases n - i\n"
            "  {\n"
            "    assert(s >= 0);  /* holds by invariant */\n"
            "    s = s + i;\n"
            "    i = i + 1;\n"
            "  }\n"
            "  s\n"
            "}\n"
            "} // verus!\n"
        )
        prog = Program(source, Language.VERUS)
        stripped = b.strip(prog)
        assert b.complexity(prog) == b.complexity(stripped)

    def test_feature_sets_invariant_to_strip(self):
        b = backend()
        source = (
            "verus! {\n"
            "fn binary_search(a: &Vec<i64>, target: i64) -> (idx: usize)\n"
            "  requires forall|i: int, j: int| 0 <= i < j < a.len() ==> a[i] <= a[j]\n"
            "  ensures idx < a.len() ==> a[idx as int] == target  // found\n"
            "{\n"
            "  /* simple linear scan for now */\n"
            "  let mut idx: usize = a.len();\n"
            "  let mut i: usize = 0;\n"
            "  while i < a.len()\n"
            "    invariant 0 <= i  // lower bound\n"
            "    invariant idx == a.len()\n"
            "  {\n"
            "    if a[i] == target { idx = i; return idx; }\n"
            "    i = i + 1;\n"
            "  }\n"
            "  idx\n"
            "}\n"
            "} // verus!\n"
        )
        prog = Program(source, Language.VERUS)
        stripped = b.strip(prog)
        assert b.feature_sets(prog) == b.feature_sets(stripped)


class TestBackendMetadata:
    """Backend registration and metric-set membership."""

    def test_complexity_metrics_nonempty(self):
        assert len(backend().complexity_metrics) > 0

    def test_feature_metrics_nonempty(self):
        assert len(backend().feature_metrics) > 0

    def test_language_enum_returns_verus_backend(self):
        assert type(Language.VERUS.get_backend()).__name__ == "VerusBackend"

    def test_file_extension(self):
        assert backend().file_extension == "rs"

    def test_declaration_keywords_nonempty(self):
        assert len(backend().declaration_keywords) > 0
