"""Shared size limits for enqueuing follow-up tasks.

Programs are re-sent in full on every `repair`/`extend` attempt, so large
programs dominate input-token cost (and, with an expensive model, run cost).
Workers use ``program_within_limit`` to skip creating extend/repair follow-up
tasks for programs that have already grown past ``max_tokens`` tokens.
"""
from typing import Optional

import tiktoken

# Default cap (tiktoken cl100k_base tokens) on a program's size for enqueuing
# extend/repair follow-up tasks. ``None`` disables the cap.
DEFAULT_MAX_PROGRAM_TOKENS: Optional[int] = 4096

_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def program_token_count(text: str) -> int:
    """Approximate token count of a program (tiktoken cl100k_base)."""
    return len(_encoder().encode(text or "", disallowed_special=()))


def program_within_limit(text: str, max_tokens: Optional[int]) -> bool:
    """Whether `text` is short enough to enqueue an extend/repair follow-up.

    Returns True (no limit) when ``max_tokens`` is None.
    """
    if max_tokens is None:
        return True
    return program_token_count(text) < max_tokens
