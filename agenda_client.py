#!/usr/bin/env python3
"""
AgendaClient worker launcher.

Runs workers that connect to a remote AgendaServer.

Usage examples:
    # Run with Hydra config
    python agenda_client.py +agenda=distributed +scheduler=distributed_example

    # Run with CLI args
    python agenda_client.py --host 127.0.0.1 --port 9999 --worker-type generator --fuel 10 --duration 60
"""

import argparse
import asyncio
import logging
import sys
import time

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os

from agenda_distributed import AgendaClient
from scheduler import Scheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


async def run_with_hydra(config_name: str):
    """Run using Hydra configuration."""
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
        agenda = instantiate(cfg.agenda)
        scheduler = instantiate(cfg.scheduler)
        await scheduler.run(agenda)


async def run_with_cli(host: str, port: int, unix_socket: str, worker_type: str, fuel: int, duration: float):
    """Run using CLI arguments."""
    from worker.dummy_workers import DummyIdeaGenerator, DummyImplementer
    from scheduler import RoundRobinScheduler

    # Connect to server
    if unix_socket:
        agenda = AgendaClient(unix_socket=unix_socket)
    else:
        agenda = AgendaClient(host=host, port=port)

    # Create worker
    if worker_type == "generator":
        worker = DummyIdeaGenerator()
    elif worker_type == "implementer":
        worker = DummyImplementer()
    else:
        raise ValueError(f"Unknown worker type: {worker_type}")

    logger.info(f"Worker {worker_type} starting...")

    # Work for specified duration (if given) or forever
    start_time = time.time()
    turns = 0

    try:
        while duration is None or (time.time() - start_time < duration):
            await worker.work(agenda, fuel)
            turns += 1
            logger.info(f"Worker {worker_type} completed turn {turns}")
            await asyncio.sleep(0.1)  # Small delay between turns
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await agenda.close()
        logger.info(f"Worker {worker_type} finished after {turns} turns")
