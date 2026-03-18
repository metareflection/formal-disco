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

    @abstractmethod
    def log_code_base_statistics(
        self,
        codebase_stats: dict[str, int],
    ) -> None:
        """Log aggregate statistics about the collection of programs we have so far."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log a flat dict of metric_name -> value verbatim, without any prefix."""
        pass

    @abstractmethod
    def log_task_outcomes(self, outcomes: dict[str, dict[str, int]]) -> None:
        """Log cumulative task outcome counts and derived success rates.

        outcomes maps task_type -> {terminal_state: count}, where terminal states
        are the WorkStatus string values (DONE, FAILED, ATTEMPTED).
        """
        pass

    @abstractmethod
    def log_performance_statistics(
        self,
        per_procedure_stats: dict[str, dict[str, float]],
        aggregate_stats: dict[str, float],
    ) -> None:
        """Log RPC performance statistics (latency, call rates).

        aggregate_stats includes 'active_clients' count.
        """
        pass

