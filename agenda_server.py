#!/usr/bin/env python3
"""
AgendaServer entry point.

Starts an AgendaServer wrapping a LocalAgenda and serves requests.

Usage examples:
    # Using config files
    python agenda_server.py agenda=local server=tcp
    python agenda_server.py agenda=local server=unix

    # With overrides
    python agenda_server.py agenda=local server.port=8888
"""

import asyncio
import logging

from hydra import main
from hydra.utils import instantiate
from omegaconf import DictConfig

from agenda_distributed import AgendaServer
from performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


@main(config_path="config", config_name="server", version_base=None)
def _main(cfg: DictConfig):
    # Create shared performance tracker for RPC latency monitoring
    perf_tracker = PerformanceTracker(window_seconds=600.0)

    # Pass tracker to agenda (for logging during checkpoint) and server (for recording calls)
    agenda = instantiate(cfg.agenda, performance_tracker=perf_tracker)
    server: AgendaServer = instantiate(cfg.server, agenda=agenda, performance_tracker=perf_tracker)

    logger.info("Starting AgendaServer...")
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    _main()
