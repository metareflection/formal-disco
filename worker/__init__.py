#!/usr/bin/env python3
"""
Worker abstraction and some example implementations.

- Worker: abstract base with a single async method `work(agenda, fuel: int)`.
- DummyIdeaGenerator: creates 'idea/{number}.txt' objects and enqueues 'implement' tasks.
- DummyImplementer: consumes 'implement' tasks, creates Dafny programs, marks tasks DONE.
"""

from abc import ABC, abstractmethod

from agenda import Agenda


class Worker(ABC):
    """
    These are the main actors in the discovery system:
    workers add and perform tasks from the agenda, and create and update objects.
    """
    @abstractmethod
    async def work(self, agenda: Agenda, fuel: int) -> None:
        """
        Perform some work against the agenda, consuming `fuel` units.

        For instance, fuel can correspond to how many tasks the worker should attempt.
        Specific semantics of "fuel" are worker-specific, but higher fuel should do more work.
        """
        raise NotImplementedError
