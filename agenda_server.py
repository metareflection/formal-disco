#!/usr/bin/env python3
"""
AgendaServer entry point.

Starts an AgendaServer wrapping a LocalAgenda and serves requests.

Usage examples:
    # TCP server
    python agenda_server.py --host 127.0.0.1 --port 9999 --checkpoint agenda.pkl

    # Unix socket server
    python agenda_server.py --unix-socket /tmp/agenda.sock --checkpoint agenda.pkl

    # With Hydra config
    python agenda_server.py +agenda=local
"""

import argparse
import asyncio
import logging
import sys

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os

from agenda import LocalAgenda
from agenda_distributed import AgendaServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run AgendaServer")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9999, help="Port to bind to")
    parser.add_argument("--unix-socket", help="Unix socket path (alternative to TCP)")
    parser.add_argument("--checkpoint", default="distributed-agenda.pkl", help="Checkpoint file path")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Checkpoint every N operations")
    parser.add_argument("--hydra-config", help="Use Hydra config name instead of CLI args")

    args = parser.parse_args()

    # Create LocalAgenda
    if args.hydra_config:
        # Use Hydra configuration
        config_dir = os.path.join(os.path.dirname(__file__), "config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=args.hydra_config)
            local_agenda = instantiate(cfg.agenda)
    else:
        # Use CLI arguments
        local_agenda = LocalAgenda(
            checkpoint_path=args.checkpoint,
            checkpoint_interval=args.checkpoint_interval,
        )

    # Create and start server
    if args.unix_socket:
        server = AgendaServer(local_agenda, unix_socket=args.unix_socket)
    else:
        server = AgendaServer(local_agenda, host=args.host, port=args.port)

    logger.info("Starting AgendaServer...")
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
