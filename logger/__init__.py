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
    def log_task_created(self, task_type: str, task_id: str) -> None:
        """Log that a new task of the given type was created."""
        pass

    @abstractmethod
    def log_task_done(self, task_type: str, task_id: str) -> None:
        """Log that a task of the given type was marked DONE."""
        pass

    @abstractmethod
    def log_task_failed(self, task_type: str, task_id: str) -> None:
        """Log that a task of the given type was marked FAILED."""
        pass

    @abstractmethod
    def log_task_assigned(self, task_type: str, task_id: str) -> None:
        """Log that a task of the given type was assigned to a worker (moved to DOING)."""
        pass

    @abstractmethod
    def log_new_working_program(self, program_text: str) -> None:
        """Log that a new working program was created (e.g., passed verification)."""
        pass
