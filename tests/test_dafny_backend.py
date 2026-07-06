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
        assert isinstance(backend().feature_sets(load("max.dfy"))["subject_word"], Counter)

    def test_subject_words_from_decl_names(self):
        words = backend().feature_sets(load("binary_search.dfy"))["subject_word"]
        # "BinarySearch" → split → ['binary', 'search']
        # 'binary' is an adjective → filtered by POS tagger; 'search' is a noun → kept
        assert "binary" not in words
        assert "search" in words

    def test_invariant_templates_extracted(self):
        templates = backend().feature_sets(load("binary_search.dfy"))["annotation_template"]
        invs = [t for t in templates if t.startswith("invariant: ")]
        assert len(invs) > 0
        for template in invs:
            assert isinstance(template, str)

    def test_ensures_templates_extracted(self):
        templates = backend().feature_sets(load("max.dfy"))["annotation_template"]
        assert any(t.startswith("ensures: ") for t in templates)

    def test_requires_templates_extracted(self):
        templates = backend().feature_sets(load("binary_search.dfy"))["annotation_template"]
        assert any(t.startswith("requires: ") for t in templates)

    def test_loop_skeletons_for_loops(self):
        skeletons = backend().feature_sets(load("binary_search.dfy"))["loop_skeleton"]
        assert len(skeletons) > 0
        assert any("while" in sk for sk in skeletons)

    def test_loop_skeletons_empty_for_no_loops(self):
        assert len(backend().feature_sets(load("max.dfy"))["loop_skeleton"]) == 0

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
        skeletons = backend().feature_sets(Program(source, Language.DAFNY))["loop_skeleton"]
        assert any("while" in sk and sk.count("while") == 2 for sk in skeletons)

    def test_template_replaces_identifiers(self):
        source = (
            "method Foo(x: int) returns (y: int)\n"
            "  ensures y == x + 1\n"
            "{ y := x + 1; }\n"
        )
        templates = backend().feature_sets(Program(source, Language.DAFNY))["annotation_template"]
        ensures = [t for t in templates if t.startswith("ensures: ")]
        assert ensures
        for tmpl in ensures:
            body = tmpl[len("ensures: "):]
            assert "x" not in body
            assert "y" not in body
            assert "*" in body

    def test_empty_program(self):
        fs = backend().feature_sets(Program("", Language.DAFNY))
        for v in fs.values():
            assert len(v) == 0


class TestNumericFeatureSets:
    """Numeric feature metrics (formerly complexity) — pure Python, no Dafny invocation."""

    def test_method_body_sizes_positive(self):
        c = backend().feature_sets(load("binary_search.dfy"))["method_body_size"]
        assert sum(c.values()) > 0
        assert all(s > 0 for s in c.elements())

    def test_annotations_per_method_for_max(self):
        # max.dfy has at least one ensures clause -> annotations_per_method records it.
        c = backend().feature_sets(load("max.dfy"))["annotations_per_method"]
        assert sum(c.values()) > 0
        assert all(n >= 1 for n in c.elements())

    def test_lemma_body_size_empty_for_non_lemma_file(self):
        # max.dfy has no lemmas, so the lemma_body_size counter is empty.
        c = backend().feature_sets(load("max.dfy"))["lemma_body_size"]
        assert len(c) == 0

    def test_language_features_detect_arrays(self):
        c = backend().feature_sets(load("binary_search.dfy"))["language_features"]
        # binary_search uses `array<T>` syntax → arrays feature should fire.
        assert c.get("arrays", 0) > 0

    def test_multiple_functions_body_sizes(self):
        c = backend().feature_sets(load("sum_array.dfy"))["method_body_size"]
        assert sum(c.values()) >= 2


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
        fs = b.feature_sets(Program(source, Language.DAFNY))
        templates = fs["annotation_template"]
        assert not any(t.startswith("assert: ") for t in templates)

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
        assert not any(t.startswith("invariant: ") for t in fs["annotation_template"])

    def test_assert_in_block_comment_not_counted(self):
        b = backend()
        source = (
            "method Foo(x: int) {\n"
            "  /* assert x > 0; */\n"
            "  var y := x;\n"
            "}\n"
        )
        fs = b.feature_sets(Program(source, Language.DAFNY))
        assert not any(t.startswith("assert: ") for t in fs["annotation_template"])

    def test_ensures_in_comment_not_counted(self):
        b = backend()
        source = (
            "// ensures result >= 0\n"
            "method Foo() returns (result: int) {\n"
            "  result := 1;\n"
            "}\n"
        )
        fs = b.feature_sets(Program(source, Language.DAFNY))
        assert not any(t.startswith("ensures: ") for t in fs["annotation_template"])

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


class TestExtractMethods:
    """Tests for extract_methods — spec clauses and line numbers."""

    def test_basic_requires_ensures(self):
        source = (
            "method Foo(x: int) returns (r: int)\n"
            "  requires x > 0\n"
            "  ensures r > x\n"
            "{\n"
            "  r := x + 1;\n"
            "}\n"
        )
        methods = backend().extract_methods(Program(source, Language.DAFNY))
        assert len(methods) == 1
        m = methods[0]
        assert m['name'] == 'Foo'
        assert m['requires'] == ['x > 0']
        assert m['ensures'] == ['r > x']

    def test_multiple_requires(self):
        source = (
            "method Bar(a: int, b: int) returns (r: int)\n"
            "  requires a > 0\n"
            "  requires b > 0\n"
            "  ensures r == a + b\n"
            "{\n"
            "  r := a + b;\n"
            "}\n"
        )
        methods = backend().extract_methods(Program(source, Language.DAFNY))
        m = methods[0]
        assert m['requires'] == ['a > 0', 'b > 0']
        assert m['ensures'] == ['r == a + b']

    def test_multiline_requires(self):
        source = (
            "method Search(a: array<int>, target: int) returns (idx: int)\n"
            "  requires forall i :: 0 <= i < a.Length\n"
            "             ==> a[i] >= 0\n"
            "  ensures idx >= 0 ==> a[idx] == target\n"
            "{\n"
            "  idx := -1;\n"
            "}\n"
        )
        methods = backend().extract_methods(Program(source, Language.DAFNY))
        m = methods[0]
        assert len(m['requires']) == 1
        assert '==>' in m['requires'][0]
        assert 'a[i] >= 0' in m['requires'][0]
        assert len(m['ensures']) == 1

    def test_no_specs(self):
        source = "method NoSpec() {\n  var x := 1;\n}\n"
        m = backend().extract_methods(Program(source, Language.DAFNY))[0]
        assert m['requires'] == []
        assert m['ensures'] == []

    def test_body_begin_end_simple(self):
        source = (
            "method Foo() {\n"   # line 0
            "  var x := 1;\n"    # line 1
            "}\n"                # line 2
        )
        m = backend().extract_methods(Program(source, Language.DAFNY))[0]
        assert m['body_begin'] == 0
        assert m['body_end'] == 2

    def test_body_begin_end_with_specs(self):
        source = (
            "// comment\n"                           # line 0
            "method Sum(n: int) returns (s: int)\n"  # line 1
            "  requires n >= 0\n"                     # line 2
            "  ensures s >= 0\n"                      # line 3
            "{\n"                                     # line 4
            "  s := 0;\n"                             # line 5
            "}\n"                                     # line 6
        )
        m = backend().extract_methods(Program(source, Language.DAFNY))[0]
        assert m['body_begin'] == 1
        assert m['body_end'] == 6

    def test_body_begin_end_multiple_methods(self):
        source = (
            "method A() {\n"              # line 0
            "  var x := 1;\n"             # line 1
            "  assert x == 1;\n"          # line 2
            "  assert x > 0;\n"           # line 3
            "}\n"                         # line 4
            "\n"                          # line 5
            "method B(n: int)\n"          # line 6
            "  requires n > 0\n"          # line 7
            "  ensures true\n"            # line 8
            "{\n"                         # line 9
            "  var y := 2;\n"             # line 10
            "}\n"                         # line 11
        )
        methods = backend().extract_methods(Program(source, Language.DAFNY))
        assert len(methods) == 2
        # A: no specs, two assertions
        assert methods[0]['body_begin'] == 0
        assert methods[0]['body_end'] == 4
        assert methods[0]['requires'] == []
        assert methods[0]['ensures'] == []
        assert methods[0]['assertions'] == 2
        # B: has specs, no assertions
        assert methods[1]['body_begin'] == 6
        assert methods[1]['body_end'] == 11
        assert methods[1]['requires'] == ['n > 0']
        assert methods[1]['ensures'] == ['true']
        assert methods[1]['assertions'] == 0

    def test_lemma_has_specs(self):
        source = (
            "lemma AddCommutes(a: int, b: int)\n"
            "  ensures a + b == b + a\n"
            "{\n"
            "}\n"
        )
        m = backend().extract_methods(Program(source, Language.DAFNY))[0]
        assert m['name'] == 'AddCommutes'
        assert m['ensures'] == ['a + b == b + a']
        assert m['requires'] == []

    def test_brace_on_same_line(self):
        source = "method Inline(x: int) requires x > 0 {\n  var y := x;\n}\n"
        m = backend().extract_methods(Program(source, Language.DAFNY))[0]
        assert m['requires'] == ['x > 0']
        assert m['body_begin'] == 0
        assert m['body_end'] == 2

    def test_multiline_ensures(self):
        source = (
            "function Interp(e: Expr): int\n"
            "  ensures Interp(e) >= 0\n"
            "         && Interp(e) < 100\n"
            "{\n"
            "  0\n"
            "}\n"
        )
        m = backend().extract_methods(Program(source, Language.DAFNY))[0]
        assert len(m['ensures']) == 1
        assert '&& Interp(e) < 100' in m['ensures'][0]

    def test_decreases_clause(self):
        source = (
            "method CountDown(n: nat)\n"
            "  requires n >= 0\n"
            "  decreases n\n"
            "{\n"
            "}\n"
        )
        m = backend().extract_methods(Program(source, Language.DAFNY))[0]
        assert m['decreases'] == ['n']
        assert m['requires'] == ['n >= 0']

    def test_reads_clause(self):
        source = (
            "function ReadArr(a: array<int>): int\n"
            "  reads a\n"
            "  requires a.Length > 0\n"
            "{\n"
            "  a[0]\n"
            "}\n"
        )
        m = backend().extract_methods(Program(source, Language.DAFNY))[0]
        assert m['reads'] == ['a']
        assert m['requires'] == ['a.Length > 0']

    def test_modifies_clause(self):
        source = (
            "method Mutate(a: array<int>)\n"
            "  requires a.Length > 0\n"
            "  modifies a\n"
            "  ensures a[0] == 42\n"
            "{\n"
            "  a[0] := 42;\n"
            "}\n"
        )
        m = backend().extract_methods(Program(source, Language.DAFNY))[0]
        assert m['modifies'] == ['a']
        assert m['requires'] == ['a.Length > 0']
        assert m['ensures'] == ['a[0] == 42']

    def test_all_spec_clauses_present_in_output(self):
        source = "method X() {\n}\n"
        m = backend().extract_methods(Program(source, Language.DAFNY))[0]
        for kw in ('requires', 'ensures', 'decreases', 'reads', 'modifies'):
            assert kw in m, f"missing key: {kw}"
            assert m[kw] == []


class TestBackendMetadata:
    """Backend registration and metric-set membership."""

    def test_feature_metrics_nonempty(self):
        assert len(backend().feature_metrics) > 0

    def test_surprisal_metrics_subset(self):
        b = backend()
        assert b.surprisal_metrics
        assert b.surprisal_metrics <= b.feature_metrics

    def test_language_enum_returns_dafny_backend(self):
        assert type(Language.DAFNY.get_backend()).__name__ == "DafnyBackend"
