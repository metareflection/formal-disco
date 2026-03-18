from prompt import (
    format_extend_user,
    format_idea_user,
    format_implement_user,
    format_repair_user,
    reconstruct_chat_messages,
)


def test_format_implement_user_deterministic():
    out = format_implement_user(idea="X")
    assert "Idea/specification:" in out
    assert "X" in out


def test_format_repair_user_contains_sections():
    out = format_repair_user(program="P", notes="N")
    assert out.startswith("Program:")
    assert "Notes (verification output):" in out
    assert "P" in out
    assert "N" in out


def test_reconstruct_user_prompt_dispatch():
    msgs = reconstruct_chat_messages("implement", {"idea": "abc"}, example_before="b", example_diff="d", example_after="a")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "abc" in msgs[1]["content"]

    msgs = reconstruct_chat_messages("extend", {"program": "p"}, example_before="b", example_diff="d", example_after="a")
    assert "Current program:" in msgs[1]["content"]

    msgs = reconstruct_chat_messages("idea", {"repo": "r", "readme": "m"}, example_before="b", example_diff="d", example_after="a")
    assert "Repository:" in msgs[1]["content"]
