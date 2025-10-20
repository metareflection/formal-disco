#!/usr/bin/env python3

"""
Logger interface for agenda events and program tracking.

Provides an abstraction for logging task lifecycle events and program statistics
to various backends (wandb, console, file, etc.).
"""

from abc import ABC, abstractmethod


class AgendaLogger(ABC):
    """
    Abstract base class for logging agenda events.

    Subclasses should implement the various event methods.
    """

    @abstractmethod
    def log_new_working_program(self, program_text: str) -> None:
        """Log that a new working program was created (e.g., passed verification)."""
        pass

    @abstractmethod
    def log_task_state(self, task_type: str, task_id: str, new_state: str) -> None:
        """Notify the logger that task `task_id` of type `task_type` entered `new_state`."""
        pass

