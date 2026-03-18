#!/usr/bin/env python3

"""
No-op logger implementation that ignores all events.
"""

from . import AgendaLogger


class NoOpLogger(AgendaLogger):
    """
    A logger that does nothing. Useful for environments where logging is not needed.
    """

    def log_new_working_program(self, program_text: str) -> None:
        """Log that a new working program was created (no-op)."""
        pass

    def log_task_state(self, task_type: str, task_id: str, new_state: str) -> None:
        """Notify the logger that task entered new_state (no-op)."""
        pass

    def log_code_base_statistics(self, codebase_stats: dict[str, int]) -> None:
        """Log aggregate statistics about programs (no-op)."""
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log a flat dict of metrics verbatim (no-op)."""
        pass

    def log_task_outcomes(self, outcomes: dict[str, dict[str, int]]) -> None:
        """Log cumulative task outcome counts and success rates (no-op)."""
        pass

    def log_performance_statistics(
        self,
        per_procedure_stats: dict[str, dict[str, float]],
        aggregate_stats: dict[str, float],
    ) -> None:
        """Log RPC performance statistics (no-op)."""
        pass