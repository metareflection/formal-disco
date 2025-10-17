#!/usr/bin/env python3
"""
No-op logger implementation that ignores all events.
"""
from . import AgendaLogger
class NoOpLogger(AgendaLogger):
    """
    A logger that does nothing. Useful for environments where logging is not needed.
    """
    def log_task_created(self, task_type: str, task_id: str) -> None:
        pass
    def log_task_done(self, task_type: str, task_id: str) -> None:
        pass
    def log_task_failed(self, task_type: str, task_id: str) -> None:
        pass
    def log_task_assigned(self, task_type: str, task_id: str) -> None:
        pass
    def log_new_working_program(self, program_text: str) -> None:
        pass
