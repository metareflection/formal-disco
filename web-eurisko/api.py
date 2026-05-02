"""FastAPI backend that reads the agenda checkpoint and serves JSON for the frontend."""

import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Eurisko Discovery Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "agenda-discovery.pkl")

_cache: dict[str, Any] = {}
_cache_time: float = 0
CACHE_TTL = 2.0  # seconds


def _load() -> dict:
    global _cache, _cache_time
    now = time.time()
    if now - _cache_time < CACHE_TTL and _cache:
        return _cache
    try:
        with open(CHECKPOINT_PATH, "rb") as f:
            _cache = pickle.load(f)
        _cache_time = now
    except FileNotFoundError:
        _cache = {"tasks": {}, "status": {}, "objects": {}, "task_outcomes": {}}
    return _cache


@app.get("/api/overview")
def overview():
    data = _load()
    tasks = data["tasks"]
    status = data["status"]
    objects = data["objects"]

    concepts = [o for o in objects.values() if o.type == "concept"]
    heuristics = [o for o in objects.values() if o.type == "heuristic"]

    status_by_type: dict[str, dict[str, int]] = {}
    for tid, t in tasks.items():
        s = status[tid]
        counts = status_by_type.setdefault(t.type, {})
        ws = str(s.work_status)
        counts[ws] = counts.get(ws, 0) + 1

    return {
        "total_tasks": len(tasks),
        "total_objects": len(objects),
        "total_concepts": len(concepts),
        "total_heuristics": len(heuristics),
        "proved_count": sum(1 for c in concepts if c.properties.get("proof_strategy")),
        "conjecture_count": sum(1 for c in concepts if c.properties.get("kind") == "conjecture"),
        "definition_count": sum(1 for c in concepts if c.properties.get("kind") in ("definition", "operation")),
        "task_status": status_by_type,
        "task_outcomes": data.get("task_outcomes", {}),
    }


@app.get("/api/concepts")
def concepts():
    data = _load()
    objects = data["objects"]
    result = []
    for o in objects.values():
        if o.type != "concept":
            continue
        p = o.properties
        result.append({
            "path": o.path,
            "name": p.get("name", ""),
            "kind": p.get("kind", ""),
            "domain": p.get("domain", ""),
            "description": p.get("description", ""),
            "lean_statement": p.get("lean_statement", ""),
            "lean_proof": p.get("lean_proof"),
            "lean_imports": p.get("lean_imports", []),
            "tags": p.get("tags", []),
            "related_concepts": p.get("related_concepts", []),
            "origin_heuristic": p.get("origin_heuristic"),
            "proof_strategy": p.get("proof_strategy"),
            "proof_attempts": p.get("proof_attempts", 0),
            "interestingness": o.interestingness,
            "parents": o.parents,
        })
    return result


@app.get("/api/heuristics")
def heuristics():
    data = _load()
    objects = data["objects"]
    result = []
    for o in objects.values():
        if o.type != "heuristic":
            continue
        p = o.properties

        # Count concepts created by this heuristic
        h_name = p.get("name", "")
        created = [
            obj for obj in objects.values()
            if obj.type == "concept" and obj.properties.get("origin_heuristic") == h_name
        ]
        proved = [c for c in created if c.properties.get("proof_strategy")]

        result.append({
            "path": o.path,
            "name": h_name,
            "heuristic_kind": p.get("heuristic_kind", ""),
            "eurisclo_origin": p.get("eurisclo_origin"),
            "input_concept_kinds": p.get("input_concept_kinds", []),
            "attempts": p.get("attempts", 0),
            "successes": p.get("successes", 0),
            "interestingness": o.interestingness,
            "born_from_reflection": p.get("born_from_reflection", False),
            "concepts_created": len(created),
            "theorems_proved": len(proved),
            "template": o.content.decode("utf-8") if o.content else "",
        })
    result.sort(key=lambda h: h["interestingness"], reverse=True)
    return result


@app.get("/api/theorems")
def theorems():
    data = _load()
    objects = data["objects"]
    result = []
    for o in objects.values():
        if o.type != "concept":
            continue
        p = o.properties
        if not p.get("proof_strategy"):
            continue
        result.append({
            "name": p.get("name", ""),
            "description": p.get("description", ""),
            "lean_statement": p.get("lean_statement", ""),
            "lean_proof": p.get("lean_proof", ""),
            "proof_strategy": p.get("proof_strategy", ""),
            "origin_heuristic": p.get("origin_heuristic"),
            "interestingness": o.interestingness,
        })
    return result


@app.get("/api/trace")
def trace(kind: Optional[str] = None, worker: Optional[str] = None, limit: int = 500):
    """Reflective-event trace.

    Reads `type=='trace'` objects from the agenda. Optionally filters by event kind
    or originating worker. Returns the most recent ``limit`` events, newest first.
    """
    data = _load()
    objects = data["objects"]
    events = []
    for o in objects.values():
        if o.type != "trace":
            continue
        p = o.properties
        if kind and p.get("kind") != kind:
            continue
        if worker and p.get("worker") != worker:
            continue
        try:
            payload = json.loads(o.content.decode("utf-8")) if o.content else {}
        except Exception:
            payload = {}
        events.append({
            "path": o.path,
            "kind": p.get("kind", ""),
            "worker": p.get("worker", ""),
            "tick": p.get("tick", 0),
            "seq": p.get("seq", 0),
            "timestamp": p.get("timestamp", ""),
            "payload": payload.get("payload", {}),
        })
    events.sort(key=lambda e: (e["tick"], e["seq"]), reverse=True)
    return events[:limit]


@app.get("/api/trace/summary")
def trace_summary():
    """Counts of trace events grouped by (kind, worker), useful for sanity checks."""
    data = _load()
    objects = data["objects"]
    by_kind: dict[str, int] = {}
    by_worker: dict[str, int] = {}
    by_kind_worker: dict[str, int] = {}
    total = 0
    for o in objects.values():
        if o.type != "trace":
            continue
        total += 1
        k = o.properties.get("kind", "?")
        w = o.properties.get("worker", "?")
        by_kind[k] = by_kind.get(k, 0) + 1
        by_worker[w] = by_worker.get(w, 0) + 1
        by_kind_worker[f"{w}/{k}"] = by_kind_worker.get(f"{w}/{k}", 0) + 1
    return {
        "total": total,
        "by_kind": by_kind,
        "by_worker": by_worker,
        "by_kind_worker": by_kind_worker,
    }


@app.get("/api/graph")
def graph():
    """Return nodes and edges for the concept graph."""
    data = _load()
    objects = data["objects"]

    nodes = []
    edges = []
    seen_edges = set()

    for o in objects.values():
        if o.type != "concept":
            continue
        p = o.properties
        nodes.append({
            "id": o.path,
            "name": p.get("name", ""),
            "kind": p.get("kind", ""),
            "origin_heuristic": p.get("origin_heuristic"),
            "interestingness": o.interestingness,
            "proved": p.get("proof_strategy") is not None,
        })

        # Parent edges
        for parent_path in o.parents:
            if parent_path in {obj.path for obj in objects.values() if obj.type == "concept"}:
                edge_key = (parent_path, o.path)
                if edge_key not in seen_edges:
                    edges.append({"source": parent_path, "target": o.path, "type": "parent"})
                    seen_edges.add(edge_key)

        # Related edges
        domain = p.get("domain", "")
        for rel_name in p.get("related_concepts", []):
            rel_path = f"concept/{domain}/{rel_name}"
            if rel_path in {obj.path for obj in objects.values()}:
                edge_key = tuple(sorted([o.path, rel_path]))
                if edge_key not in seen_edges:
                    edges.append({"source": o.path, "target": rel_path, "type": "related"})
                    seen_edges.add(edge_key)

    return {"nodes": nodes, "edges": edges}


# Serve frontend static files if built
dist_dir = Path(__file__).parent / "dist"
if dist_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")
