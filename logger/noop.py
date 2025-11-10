#!/usr/bin/env python3

"""
No-op logger implementation that ignores all events.
"""

from . import AgendaLogger


class NoOpLogger(AgendaLogger):
    """
    A logger that does nothing. Useful for environments where logging is not needed.
    """