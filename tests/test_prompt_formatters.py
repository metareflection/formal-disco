from language import Language

_backend = Language.DAFNY.get_backend()
_pb = _backend.prompt_builder


def test_implement_prompt():
    msgs = _pb.implement(idea="X")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "X" in msgs[1]["content"]


def test_repair_prompt():
    msgs = _pb.repair(program="P", notes="N", example_before="b", example_diff="d", example_after="a")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "P" in msgs[1]["content"]
    assert "N" in msgs[1]["content"]


def test_extend_prompt():
    msgs = _pb.extend(program="P", example_before="b", example_diff="d", example_after="a")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "P" in msgs[1]["content"]


def test_idea_prompt():
    msgs = _pb.idea(repo="my-repo", readme="Some README text")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "my-repo" in msgs[1]["content"]
    assert "Some README text" in msgs[1]["content"]
