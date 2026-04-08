#!/usr/bin/env python3

from patch import compute_text_diff, apply_text_diff


def test_ambiguous_diff():

    p1 = """
if x {
    a := 0;
}
if x {
    b := 0;
}
"""

    p2 = """
if x {
    a := 0;
}
if x {
   assert a == 0;
   b := 0;
}
"""

    # The diff should be able to locate the right place to insert the assertion,
    # even if the previous line is the same.
    diff = compute_text_diff(p1, p2)
    assert apply_text_diff(p1, diff) == p2


def test_simple_insert():
    before = "a\nb\nc\n"
    after = "a\nb\nx\nc\n"
    diff = compute_text_diff(before, after)
    assert apply_text_diff(before, diff) == after


def test_simple_delete():
    before = "a\nb\nc\n"
    after = "a\nc\n"
    diff = compute_text_diff(before, after)
    assert apply_text_diff(before, diff) == after


def test_simple_replace():
    before = "a\nb\nc\n"
    after = "a\nB\nc\n"
    diff = compute_text_diff(before, after)
    assert apply_text_diff(before, diff) == after


def test_insert_at_beginning():
    before = "a\nb\n"
    after = "x\na\nb\n"
    diff = compute_text_diff(before, after)
    assert apply_text_diff(before, diff) == after


def test_multiple_disjoint_changes():
    """Multiple change blocks separated by unchanged regions."""
    before = (
        "method A() {\n"
        "  var x := 1;\n"
        "}\n"
        "\n"
        "method B() {\n"
        "  var y := 2;\n"
        "}\n"
        "\n"
        "method C() {\n"
        "  var z := 3;\n"
        "}\n"
    )
    after = (
        "method A() {\n"
        "  var x := 1;\n"
        "  assert x > 0;\n"
        "}\n"
        "\n"
        "method B() {\n"
        "  var y := 2;\n"
        "}\n"
        "\n"
        "method C() {\n"
        "  var z := 3;\n"
        "  assert z > 0;\n"
        "}\n"
    )
    diff = compute_text_diff(before, after)
    assert apply_text_diff(before, diff) == after


def test_ambiguous_repeated_blocks():
    """Three identical if-blocks, change only in the last one."""
    before = (
        "if x {\n"
        "  a := 0;\n"
        "}\n"
        "if x {\n"
        "  a := 0;\n"
        "}\n"
        "if x {\n"
        "  a := 0;\n"
        "}\n"
    )
    after = (
        "if x {\n"
        "  a := 0;\n"
        "}\n"
        "if x {\n"
        "  a := 0;\n"
        "}\n"
        "if x {\n"
        "  a := 0;\n"
        "  b := 1;\n"
        "}\n"
    )
    diff = compute_text_diff(before, after)
    assert apply_text_diff(before, diff) == after


def test_ambiguous_with_delete_and_insert():
    """Repeated structure with both deletion and insertion in one block."""
    before = (
        "while i < n\n"
        "  invariant i >= 0\n"
        "{\n"
        "  i := i + 1;\n"
        "}\n"
        "while i < n\n"
        "  invariant i >= 0\n"
        "{\n"
        "  i := i + 2;\n"
        "}\n"
    )
    after = (
        "while i < n\n"
        "  invariant i >= 0\n"
        "{\n"
        "  i := i + 1;\n"
        "}\n"
        "while i < n\n"
        "  invariant i >= 0\n"
        "  invariant i <= n\n"
        "{\n"
        "  assert i < n;\n"
        "  i := i + 2;\n"
        "}\n"
    )
    diff = compute_text_diff(before, after)
    assert apply_text_diff(before, diff) == after


def test_longer_program_multiple_changes():
    """Realistic Dafny-like program with multiple disjoint edits."""
    before = (
        "method Sum(a: array<int>) returns (s: int)\n"
        "  requires a.Length > 0\n"
        "  ensures s >= 0\n"
        "{\n"
        "  s := 0;\n"
        "  var i := 0;\n"
        "  while i < a.Length\n"
        "  {\n"
        "    s := s + a[i];\n"
        "    i := i + 1;\n"
        "  }\n"
        "}\n"
        "\n"
        "method Max(a: array<int>) returns (m: int)\n"
        "  requires a.Length > 0\n"
        "  ensures m >= 0\n"
        "{\n"
        "  m := a[0];\n"
        "  var i := 1;\n"
        "  while i < a.Length\n"
        "  {\n"
        "    if a[i] > m {\n"
        "      m := a[i];\n"
        "    }\n"
        "    i := i + 1;\n"
        "  }\n"
        "}\n"
    )
    after = (
        "method Sum(a: array<int>) returns (s: int)\n"
        "  requires a.Length > 0\n"
        "  ensures s >= 0\n"
        "{\n"
        "  s := 0;\n"
        "  var i := 0;\n"
        "  while i < a.Length\n"
        "    invariant 0 <= i <= a.Length\n"
        "  {\n"
        "    s := s + a[i];\n"
        "    i := i + 1;\n"
        "  }\n"
        "}\n"
        "\n"
        "method Max(a: array<int>) returns (m: int)\n"
        "  requires a.Length > 0\n"
        "  ensures m >= 0\n"
        "{\n"
        "  m := a[0];\n"
        "  var i := 1;\n"
        "  while i < a.Length\n"
        "    invariant 1 <= i <= a.Length\n"
        "  {\n"
        "    if a[i] > m {\n"
        "      m := a[i];\n"
        "    }\n"
        "    i := i + 1;\n"
        "  }\n"
        "}\n"
    )
    diff = compute_text_diff(before, after)
    assert apply_text_diff(before, diff) == after


def test_delete_multiple_lines():
    """Remove several consecutive lines from the middle."""
    before = "a\nb\nc\nd\ne\n"
    after = "a\ne\n"
    diff = compute_text_diff(before, after)
    assert apply_text_diff(before, diff) == after


def test_identical_files():
    """No changes should produce an empty diff."""
    text = "a\nb\nc\n"
    diff = compute_text_diff(text, text)
    assert diff == ""
    assert apply_text_diff(text, diff) == text
