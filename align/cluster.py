"""Synonym clustering via transitive closure of iff alignments.

After running pairwise iff probes within an invented-concept pool, group
concepts into equivalence classes: two concepts are in the same cluster if
the system has a chain of iff-proofs connecting them.

Used for the "invented-vs-invented" mode of `align_run.py`, which catches
cases like multiple LLM-named variants of the same predicate
(e.g. `is_loopless_matroid` vs `loopless_matroid_on_ground`).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import AlignmentResult


@dataclass
class SynonymCluster:
    representative: str
    members: list[str] = field(default_factory=list)
    # Each entry: (a, b, tactic_name) — proof that a iff b
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


class _UnionFind:
    def __init__(self):
        self._parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x

    def find(self, x: str) -> str:
        # Path compression
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        cur = x
        while self._parent[cur] != root:
            self._parent[cur], cur = root, self._parent[cur]
        return root

    def union(self, x: str, y: str) -> None:
        self.add(x); self.add(y)
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            # Pick the lexicographically smaller root for stable representative
            if rx < ry:
                self._parent[ry] = rx
            else:
                self._parent[rx] = ry

    def members(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for x in self._parent:
            groups[self.find(x)].append(x)
        return dict(groups)


def cluster_synonyms(results: list[AlignmentResult]) -> list[SynonymCluster]:
    """Group invented concepts into synonym clusters.

    `results` should come from a self-alignment run (canonical pool ==
    invented pool minus self). Two concepts a, b are merged if there exists
    an AlignmentResult for a with a match against b (or vice versa).
    """
    uf = _UnionFind()
    edges: list[tuple[str, str, str]] = []
    for r in results:
        uf.add(r.invented_name)
        for m in r.matches:
            uf.union(r.invented_name, m.canonical_name)
            edges.append((r.invented_name, m.canonical_name, m.tactic_name))

    clusters: list[SynonymCluster] = []
    for rep, members in uf.members().items():
        if len(members) < 2:
            continue  # singletons aren't synonym clusters
        cluster_edges = [(a, b, t) for (a, b, t) in edges
                         if a in members and b in members]
        clusters.append(SynonymCluster(
            representative=rep,
            members=sorted(members),
            edges=cluster_edges,
        ))
    # Largest cluster first
    clusters.sort(key=lambda c: -c.size)
    return clusters
