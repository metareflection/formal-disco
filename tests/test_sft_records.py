import json
import pickle
from pathlib import Path

from distill import build_sft_records


class DummyObj:
    def __init__(self, content: bytes):
        self.content = content


def _ex(prompt: str, outcome: str) -> bytes:
    return json.dumps(
        {
            "prompt": prompt,
            "arguments": {"idea": "do x", "program": "p", "notes": "n", "repo": "r", "readme": "m"},
            "response": "y",
            "outcome": outcome,
        },
        ensure_ascii=False,
    ).encode("utf-8")


def test_build_sft_records_success_only(tmp_path: Path):
    data = {
        "objects": {
            "distil/a.json": DummyObj(_ex("implement", "success")),
            "distil/b.json": DummyObj(_ex("implement", "fail")),
        }
    }
    pkl = tmp_path / "agenda.pkl"
    pkl.write_bytes(pickle.dumps(data))

    recs, counts = build_sft_records(pickle_paths=[pkl], success_only=True)
    assert len(recs) == 1
    assert "prompt" in recs[0] and "completion" in recs[0]


def test_build_sft_records_all_outcomes(tmp_path: Path):
    data = {
        "objects": {
            "distil/a.json": DummyObj(_ex("implement", "success")),
            "distil/b.json": DummyObj(_ex("implement", "fail")),
        }
    }
    pkl = tmp_path / "agenda.pkl"
    pkl.write_bytes(pickle.dumps(data))

    recs, counts = build_sft_records(pickle_paths=[pkl], success_only=False)
    assert len(recs) == 2


def test_build_sft_records_multiple_pickles(tmp_path: Path):
    data1 = {
        "objects": {
            "distil/a.json": DummyObj(_ex("implement", "success")),
        }
    }
    data2 = {
        "objects": {
            "distil/b.json": DummyObj(_ex("implement", "success")),
        }
    }

    pkl1 = tmp_path / "agenda1.pkl"
    pkl2 = tmp_path / "agenda2.pkl"
    pkl1.write_bytes(pickle.dumps(data1))
    pkl2.write_bytes(pickle.dumps(data2))

    recs, counts = build_sft_records(pickle_paths=[pkl1, pkl2], success_only=True)
    assert len(recs) == 2
