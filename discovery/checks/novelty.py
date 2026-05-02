"""Novelty check via embedding similarity.

A pre-proof gate that rejects candidates whose statement is too close (in
embedding cosine similarity) to existing entries: the agenda's already-admitted
concepts and the sibling LeanDisco library's `.lean` files.

This is what stops the system from "discovering" things that already live in
its own corpus. Mathlib coverage is a Phase 2 stretch goal; for Phase 1 the
index covers what the system has produced so far + what's been written to
LeanDisco.

Implementation notes
--------------------
- Uses HuggingFace ``transformers`` directly (already a dep via distill.py),
  with ``sentence-transformers/all-MiniLM-L6-v2`` as the default. Mean-pool
  the last hidden state, L2-normalize.
- The corpus and embeddings are cached to a ``.novelty.npz`` next to the
  agenda checkpoint, with a content-hash key so we recompute when the corpus
  changes by more than a small delta.
- Inference is synchronous; on CPU it's ~50ms per query, dwarfed by the
  Lean compile that follows.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from discovery.checks import CheckResult

logger = logging.getLogger(__name__)


@dataclass
class CorpusEntry:
    label: str  # human-readable identifier ("concept/group/abelian", "leandisco/MulComm.lean")
    text: str   # the statement text used for the embedding


def _statement_signature(text: str) -> str:
    """Heuristic 'just the theorem statement' extraction.

    Strips comments, blank lines, and proof bodies. Keeps everything up to
    ``:=`` and falls back to the whole text if no separator is present.
    """
    cleaned = re.sub(r"--[^\n]*", "", text)
    cleaned = re.sub(r"/-.*?-/", "", cleaned, flags=re.DOTALL)
    head = cleaned.split(":=", 1)[0]
    head = "\n".join(l for l in head.splitlines() if l.strip())
    return head.strip() or cleaned.strip()


def collect_agenda_corpus(agenda) -> list[CorpusEntry]:
    """Pull all concepts off the agenda as corpus entries.

    Uses the private ``_objects`` dict because the public ``Agenda`` protocol
    only exposes ``get_object(path)`` (no enumeration). Treat this as a Phase 1
    expedient — Phase 2 redesign should add a proper query API.
    """
    objects = getattr(agenda, "_objects", None)
    if not objects:
        return []
    out: list[CorpusEntry] = []
    for path, obj in objects.items():
        if obj.type != "concept":
            continue
        stmt = obj.properties.get("lean_statement", "")
        if not stmt:
            continue
        out.append(CorpusEntry(label=path, text=_statement_signature(stmt)))
    return out


def collect_leandisco_corpus(lake_project_dir: Optional[str], max_files: int = 5000) -> list[CorpusEntry]:
    """Walk LeanDisco's Domains/ tree and pull statements out of each .lean file."""
    if not lake_project_dir:
        return []
    root = Path(lake_project_dir) / "LeanDisco" / "Domains"
    if not root.is_dir():
        return []
    out: list[CorpusEntry] = []
    for path in sorted(root.rglob("*.lean")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        sig = _statement_signature(text)
        if not sig:
            continue
        rel = path.relative_to(root)
        out.append(CorpusEntry(label=f"leandisco/{rel}", text=sig))
        if len(out) >= max_files:
            break
    return out


class NoveltyIndex:
    """Embedding index over the agenda + LeanDisco corpus.

    Lazily loads the embedding model. Caches embeddings to disk keyed by a
    SHA-256 of the corpus texts so we recompute only when the corpus changes.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        cache_path: Optional[str] = None,
        device: str = "cpu",
        max_length: int = 128,
    ) -> None:
        self._model_name = model_name
        self._cache_path = cache_path
        self._device = device
        self._max_length = max_length
        self._model = None
        self._tokenizer = None
        self._labels: list[str] = []
        self._embeddings: Optional[np.ndarray] = None  # [N, D], L2-normalized

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer  # noqa: WPS433

        logger.info("Loading novelty embedding model: %s", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()

    def _encode(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        import torch  # noqa: WPS433

        with torch.no_grad():
            toks = self._tokenizer(
                texts, padding=True, truncation=True,
                max_length=self._max_length, return_tensors="pt",
            ).to(self._device)
            out = self._model(**toks)
            mask = toks["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            norm = pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            normalized = pooled / norm
        return normalized.cpu().numpy().astype(np.float32)

    @staticmethod
    def _corpus_key(entries: Iterable[CorpusEntry]) -> str:
        h = hashlib.sha256()
        for e in entries:
            h.update(e.label.encode("utf-8", "ignore"))
            h.update(b"\x00")
            h.update(e.text.encode("utf-8", "ignore"))
            h.update(b"\x00\x00")
        return h.hexdigest()

    def build(self, entries: list[CorpusEntry]) -> None:
        """(Re)build the index from a corpus, with disk cache."""
        if not entries:
            self._labels = []
            self._embeddings = np.zeros((0, 1), dtype=np.float32)
            return

        key = self._corpus_key(entries)
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                cached = np.load(self._cache_path, allow_pickle=True)
                if str(cached.get("key", "")) == key:
                    self._labels = list(cached["labels"])
                    self._embeddings = cached["embeddings"]
                    logger.info("Novelty index loaded from cache (%d entries)",
                                len(self._labels))
                    return
            except Exception as e:  # noqa: BLE001
                logger.debug("Novelty cache miss / load error: %s", e)

        logger.info("Encoding novelty corpus: %d entries", len(entries))
        texts = [e.text for e in entries]
        # Encode in batches to keep memory bounded.
        batch_size = 64
        embs = []
        for i in range(0, len(texts), batch_size):
            embs.append(self._encode(texts[i : i + batch_size]))
        embeddings = np.concatenate(embs, axis=0)

        self._labels = [e.label for e in entries]
        self._embeddings = embeddings

        if self._cache_path:
            try:
                os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
                np.savez_compressed(
                    self._cache_path,
                    key=key,
                    labels=np.array(self._labels, dtype=object),
                    embeddings=embeddings,
                )
                logger.info("Novelty index cached to %s", self._cache_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to cache novelty index: %s", e)

    def nearest(self, statement: str) -> tuple[str, float]:
        """Return ``(nearest_label, cosine_similarity)`` for ``statement``.

        Returns ``("", 0.0)`` when the index is empty — the candidate then
        passes by default.
        """
        if self._embeddings is None or len(self._labels) == 0:
            return ("", 0.0)
        emb = self._encode([_statement_signature(statement)])  # [1, D]
        sims = emb @ self._embeddings.T  # [1, N]
        idx = int(sims.argmax())
        return (self._labels[idx], float(sims[0, idx]))

    def add(self, label: str, statement: str) -> None:
        """Append a new entry to the index in memory (no cache write).

        Use after admitting a candidate so subsequent novelty checks see it.
        """
        emb = self._encode([_statement_signature(statement)])  # [1, D]
        if self._embeddings is None or len(self._labels) == 0:
            self._embeddings = emb
        else:
            self._embeddings = np.concatenate([self._embeddings, emb], axis=0)
        self._labels.append(label)


def check_novelty(
    *,
    statement: str,
    index: NoveltyIndex,
    threshold: float = 0.92,
) -> CheckResult:
    """Reject candidates too close to existing entries by cosine similarity.

    The threshold defaults to 0.92 — high enough that genuine paraphrases /
    sub-cases pass while literal restatements get caught. Tune in config once
    we have data.
    """
    if not statement:
        return CheckResult(passed=True, verdict="novel", reason="empty statement")
    label, score = index.nearest(statement)
    if score > threshold:
        return CheckResult(
            passed=False,
            verdict="too_similar",
            reason=f"cosine={score:.3f} to {label}",
            witness=label,
            details={"similarity": score, "nearest": label},
        )
    return CheckResult(
        passed=True,
        verdict="novel",
        reason=f"cosine={score:.3f} (nearest: {label})",
        details={"similarity": score, "nearest": label},
    )
