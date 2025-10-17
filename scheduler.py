#!/usr/bin/env python3
"""
Schedulers for workers. This is the main entry point for running things.

Usage example (actual configs are in config/):
    python scheduler.py agenda=local scheduler=example
"""

import asyncio
import logging
from typing import Protocol

from hydra import main
from hydra.utils import instantiate
from omegaconf import DictConfig

from agenda import Agenda
from worker import Worker

logger = logging.getLogger(__name__)


class Scheduler(Protocol):
    async def run(self, agenda: Agenda):
        raise NotImplementedError


class RoundRobinScheduler:
    """
    Simple scheduler that cycles through workers, giving each `turn_fuel` units of fuel per turn.
    """

    def __init__(self, turn_fuel: int, workers: list[Worker]) -> None:
        self.turn_fuel = max(0, int(turn_fuel))
        self.workers = list(workers)

    async def run(self, agenda: Agenda) -> None:
        idx = 0
        while True:
            worker = self.workers[idx]
            logger.info(f"Scheduling {type(worker).__name__} to work.")
            await worker.work(agenda, self.turn_fuel)
            logger.info(f"Worker {type(worker).__name__} finished a turn.")
            idx = (idx + 1) % len(self.workers)


@main(config_path="config", config_name=None, version_base=None)
def _main(cfg: DictConfig) -> None:
    agenda: Agenda = instantiate(cfg.agenda)
    scheduler: Scheduler = instantiate(cfg.scheduler)
    asyncio.run(scheduler.run(agenda))


if __name__ == "__main__":
    _main()
