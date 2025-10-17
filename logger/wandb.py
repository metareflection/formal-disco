#!/usr/bin/env python3

"""
Weights & Biases logger implementation for tracking agenda metrics.
"""

from . import AgendaLogger


class WandbLogger(AgendaLogger):
    """
    Logger that tracks and logs metrics to Weights & Biases (wandb).

    Maintains counters for each task type (created, done, failed, assigned) and
    aggregates statistics on working programs.
    """

    def __init__(self, project: str = "formal-disco") -> None:
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise RuntimeError("wandb is required to use WandbLogger; install it with `pip install wandb`")

        # Initialize wandb project
        self._wandb.init(project=project)

        # Counters per task type: {task_type: {metric: count}}
        self._counters: dict[str, dict[str, int]] = {}

        # Program statistics
        self._program_count = 0
        self._program_total_lines = 0

    def log_task_created(self, task_type: str, task_id: str) -> None:
        """Increment task creation counter for the task type."""
        if task_type not in self._counters:
            self._counters[task_type] = {"created": 0, "done": 0, "failed": 0, "assigned": 0}
        self._counters[task_type]["created"] += 1
        self._wandb.log({f"task/{task_type}/created": self._counters[task_type]["created"]})

    def log_task_done(self, task_type: str, task_id: str) -> None:
        """Increment task done counter for the task type."""
        if task_type not in self._counters:
            self._counters[task_type] = {"created": 0, "done": 0, "failed": 0, "assigned": 0}
        self._counters[task_type]["done"] += 1
        self._wandb.log({f"task/{task_type}/done": self._counters[task_type]["done"]})

    def log_task_failed(self, task_type: str, task_id: str) -> None:
        """Increment task failed counter for the task type."""
        if task_type not in self._counters:
            self._counters[task_type] = {"created": 0, "done": 0, "failed": 0, "assigned": 0}
        self._counters[task_type]["failed"] += 1
        self._wandb.log({f"task/{task_type}/failed": self._counters[task_type]["failed"]})

    def log_task_assigned(self, task_type: str, task_id: str) -> None:
        """Increment task assigned counter for the task type."""
        if task_type not in self._counters:
            self._counters[task_type] = {"created": 0, "done": 0, "failed": 0, "assigned": 0}
        self._counters[task_type]["assigned"] += 1
        self._wandb.log({f"task/{task_type}/assigned": self._counters[task_type]["assigned"]})

    def log_new_working_program(self, program_text: str) -> None:
        """Log a new working program: increment count and add line count."""
        self._program_count += 1
        num_lines = len([line for line in program_text.split('\n') if line.strip()])
        self._program_total_lines += num_lines

        self._wandb.log({
            "program/count": self._program_count,
            "program/total_lines": self._program_total_lines,
            "program/avg_lines": self._program_total_lines / self._program_count if self._program_count > 0 else 0,
        })
