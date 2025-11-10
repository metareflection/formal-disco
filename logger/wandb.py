#!/usr/bin/env python3

"""
Weights & Biases logger implementation for tracking agenda metrics.
"""

import collections

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


        # Track unique task ids seen per (task_type, state)
        # { task_type: { state: set(task_id) } }
        self._unique_ids: dict[tuple[str, str], set[str]] = collections.defaultdict(set)

        # Track current state per task_id.
        self._task_state: dict[str, str] = {}

        # Program statistics
        self._program_count = 0
        self._program_total_lines = 0


    def log_task_state(self, task_type: str, task_id: str, new_state: str) -> None:
        """Update internal per-state sets and log the current counts for the task type.

        This method will add the task_id to the set for `new_state` and remove it
        from any other state sets so that the per-state counts represent the
        number of tasks currently in each state.
        """
        # Remove task_id from its current state, if we know it.
        current_state = self._task_state.get(task_id)

        if current_state == new_state:
            # Nothing to do.
            return

        # Update current state mapping
        self._task_state[task_id] = new_state

        if current_state is not None:
            self._unique_ids[(task_type, current_state)].discard(task_id)
            self._wandb.log({f"task/{task_type}/current/{current_state}":
                             len(self._unique_ids[(task_type, current_state)])})

        # Add to new state set and log current counts
        self._unique_ids[(task_type, new_state)].add(task_id)
        self._wandb.log({f"task/{task_type}/current/{new_state}":
                         len(self._unique_ids[(task_type, new_state)])})

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

    def log_code_base_statistics(
        self,
        codebase_stats: dict[str, int],
    ) -> None:
        """Log aggregate statistics about the collection of programs we have so far."""
        wandb_stats = {f"codebase/{key}": value for key, value in codebase_stats.items()}
        self._wandb.log(wandb_stats)
