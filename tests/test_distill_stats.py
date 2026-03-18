import json
import pickle
from pathlib import Path

from distill import compute_distil_stats


class DummyObj:
    """Pickleable minimal object matching agenda.Object for tests (only needs .content)."""

    def __init__(self, content: bytes):
        self.content = content


def test_compute_distil_stats_basic(tmp_path: Path):
    def ex(prompt: str, outcome: str) -> bytes:
        return json.dumps(
            {
                "prompt": prompt,
                "arguments": {"x": 1},
                "response": "y",
                "outcome": outcome,
            },
            ensure_ascii=False,
        ).encode("utf-8")

    data = {
        "objects": {
            "distil/example.json": DummyObj(ex("implement", "success")),
            "distil/example_abc.json": DummyObj(ex("implement", "fail")),
            "distil/example_2.json": DummyObj(ex("repair", "success")),
            "programs/p.dfy": DummyObj(b"not json"),
            "distil/bad.json": DummyObj(b"{not json"),
        }
    }

    pkl = tmp_path / "agenda.pkl"
    pkl.write_bytes(pickle.dumps(data))

    stats = compute_distil_stats(pkl)

    assert stats.total == 3
    assert stats.outcomes["success"] == 2
    assert stats.outcomes["fail"] == 1

    assert stats.by_prompt_total["implement"] == 2
    assert stats.by_prompt_total["repair"] == 1

    assert stats.by_prompt_outcomes["implement"]["success"] == 1
    assert stats.by_prompt_outcomes["implement"]["fail"] == 1
    assert stats.by_prompt_outcomes["repair"]["success"] == 1
