"""Reflective-event trace.

A ``Tracer`` records reflective decisions to two sinks:

1. **In-agenda objects** with ``type='trace'`` and path
   ``trace/{tick:08d}/{seq:06d}/{worker}/{kind}``. These survive checkpointing,
   can be queried by the existing object store, and are surfaced by the web UI.
2. **JSONL audit log** appended live next to the checkpoint
   (``<checkpoint_basename>.trace.jsonl``). Cheap to grep/jq, durable across
   crashes, and can grow much bigger than what we'd want to keep in memory.

The trace module is intentionally side-channel: workers call ``Tracer`` from
their existing logic, and trace writes are best-effort (never raise).

Event kinds (convenience methods provided; arbitrary kinds via ``event``):

- ``heuristic_apply``: a heuristic was selected and applied to a target concept.
- ``admit`` / ``reject``: a candidate produced by a heuristic passed/failed a check.
- ``worth_update``: an interestingness/worth value changed and why.
- ``heuristic_birth`` / ``heuristic_death``: ReflectionWorker created or killed a heuristic.
- ``proof_attempt``: ProofWorker tried to prove a conjecture (success or fail).
- ``reflection``: ReflectionWorker prompt/response pair.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
from typing import Any, Optional

from agenda import Agenda, Object


class Tracer:
    """Per-worker tracer. Cheap to instantiate; safe across asyncio coroutines."""

    _seq_lock = threading.Lock()
    _seq = 0

    def __init__(self, *, agenda: Agenda, worker: str, jsonl_path: Optional[str] = None) -> None:
        self._agenda = agenda
        self._worker = worker
        if jsonl_path is None:
            ckpt = getattr(agenda, "_checkpoint_path", None)
            if ckpt:
                base, _, _ = ckpt.rpartition(".")
                jsonl_path = (base or ckpt) + ".trace.jsonl"
        self._jsonl_path = jsonl_path

    @classmethod
    def _next_seq(cls) -> int:
        with cls._seq_lock:
            cls._seq += 1
            return cls._seq

    async def event(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        parents: Optional[list[str]] = None,
    ) -> str:
        """Record one reflective event. Returns the trace object path."""
        tick = getattr(self._agenda, "_clock", 0)
        seq = self._next_seq()
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        path = f"trace/{tick:08d}/{seq:06d}/{self._worker}/{kind}"
        record = {
            "path": path,
            "kind": kind,
            "worker": self._worker,
            "tick": tick,
            "seq": seq,
            "timestamp": ts,
            "payload": _jsonable(payload),
        }
        self._write_jsonl(record)
        await self._write_object(record, path, kind, tick, seq, ts, parents)
        return path

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if not self._jsonl_path:
            return
        try:
            os.makedirs(os.path.dirname(self._jsonl_path) or ".", exist_ok=True)
            line = json.dumps(record, separators=(",", ":"), default=str)
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    async def _write_object(
        self,
        record: dict[str, Any],
        path: str,
        kind: str,
        tick: int,
        seq: int,
        ts: str,
        parents: Optional[list[str]],
    ) -> None:
        try:
            scalar_props = {
                k: v
                for k, v in record["payload"].items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }
            obj = Object(
                path=path,
                type="trace",
                parents=parents or [],
                content=json.dumps(record, default=str).encode("utf-8"),
                properties={
                    "kind": kind,
                    "worker": self._worker,
                    "tick": tick,
                    "seq": seq,
                    "timestamp": ts,
                    **scalar_props,
                },
                interestingness=1.0,
            )
            await self._agenda.create_object(obj)
        except Exception:
            pass

    async def heuristic_apply(
        self, *, heuristic: str, target: str, n_candidates: int, **extra: Any
    ) -> str:
        return await self.event(
            "heuristic_apply",
            {"heuristic": heuristic, "target": target, "n_candidates": n_candidates, **extra},
        )

    async def admit(
        self, *, heuristic: str, candidate: str, reason: str = "passed_typecheck", **extra: Any
    ) -> str:
        return await self.event(
            "admit",
            {"heuristic": heuristic, "candidate": candidate, "reason": reason, **extra},
        )

    async def reject(
        self, *, heuristic: str, candidate: str, reason: str, **extra: Any
    ) -> str:
        return await self.event(
            "reject",
            {"heuristic": heuristic, "candidate": candidate, "reason": reason, **extra},
        )

    async def worth_update(
        self, *, heuristic: str, before: float, after: float, cause: str, **extra: Any
    ) -> str:
        return await self.event(
            "worth_update",
            {"heuristic": heuristic, "before": before, "after": after, "cause": cause, **extra},
        )

    async def heuristic_birth(
        self, *, name: str, parent_heuristic: str, template: str, **extra: Any
    ) -> str:
        return await self.event(
            "heuristic_birth",
            {
                "name": name,
                "parent_heuristic": parent_heuristic,
                "template": template[:4000],
                **extra,
            },
        )

    async def heuristic_death(
        self, *, name: str, attempts: int, successes: int, **extra: Any
    ) -> str:
        return await self.event(
            "heuristic_death",
            {"name": name, "attempts": attempts, "successes": successes, **extra},
        )

    async def proof_attempt(
        self, *, theorem: str, strategy: str, outcome: str, **extra: Any
    ) -> str:
        return await self.event(
            "proof_attempt",
            {"theorem": theorem, "strategy": strategy, "outcome": outcome, **extra},
        )

    async def reflection(
        self, *, target: str, prompt: str, response: str, **extra: Any
    ) -> str:
        return await self.event(
            "reflection",
            {
                "target": target,
                "prompt": prompt[:4000],
                "response": response[:4000],
                **extra,
            },
        )


def _jsonable(o: Any) -> Any:
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    return str(o)
