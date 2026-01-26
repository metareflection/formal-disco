#!/usr/bin/env python3

"""
Distributed Agenda implementation using client-server architecture.

AgendaServer wraps a LocalAgenda and serves requests over any stream transport.
AgendaClient implements the Agenda protocol by forwarding requests to the server.

Both support TCP connections, Unix domain sockets, and os.pipe().
"""

import asyncio
import atexit
import json
import logging
import os
import pickle
import signal
import socket
import sys
import time
from typing import Any, Optional, Union

from agenda import Agenda, LocalAgenda, Object, Task, TaskStatus, WorkStatus
from performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

# Protocol: Simple pickle-based RPC over any byte stream

async def _read_message(reader: asyncio.StreamReader) -> dict:
    """Read a length-prefixed pickled message from a stream."""
    length_bytes = await reader.readexactly(4)
    msg_length = int.from_bytes(length_bytes, byteorder='big')
    msg_bytes = await reader.readexactly(msg_length)
    return pickle.loads(msg_bytes)


async def _write_message(writer: asyncio.StreamWriter, msg: dict) -> None:
    """Write a length-prefixed pickled message to a stream."""
    msg_bytes = pickle.dumps(msg)
    writer.write(len(msg_bytes).to_bytes(4, byteorder='big'))
    writer.write(msg_bytes)
    await writer.drain()


class AgendaServer:
    """
    Server that wraps a LocalAgenda and handles remote requests.

    Manages checkpointing both periodically and on termination signals.

    Can serve over TCP, Unix sockets, or communicate via pipes.
    """

    def __init__(
        self,
        agenda: LocalAgenda,
        host: Optional[str] = "127.0.0.1",
        port: Optional[int] = 9999,
        unix_socket: Optional[str] = None,
        server_address_path: Optional[str] = None,
        performance_tracker: Optional[PerformanceTracker] = None,
    ):
        self.agenda = agenda
        self.host = host
        self.port = port
        self.unix_socket = unix_socket
        self.server_address_path = server_address_path
        self.server: Optional[asyncio.Server] = None
        self._running = False
        self._performance_tracker = performance_tracker
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown and checkpointing."""
        # Register checkpoint on exit
        atexit.register(self._sync_checkpoint)

        # Setup signal handlers for SIGTERM and SIGINT
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals by checkpointing and exiting."""
        logger.info(f"Received signal {signum}, checkpointing and shutting down...")
        self._sync_checkpoint()
        sys.exit(0)

    def _sync_checkpoint(self):
        """Synchronous checkpoint for signal handlers and atexit."""
        if hasattr(self.agenda, '_checkpoint'):
            self.agenda._checkpoint()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single client connection over any stream transport."""
        addr = writer.get_extra_info('peername', 'unknown')
        # Convert address to string for tracking (tuple for TCP, string for Unix)
        client_id = str(addr) if addr else "unknown"
        logger.debug(f"Client connected from {client_id}")

        try:
            while True:
                msg = await _read_message(reader)
                response = await self.process_request(msg, client_id)
                await _write_message(writer, response)

        except asyncio.IncompleteReadError:
            logger.debug(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}", exc_info=True)
        finally:
            writer.close()
            await writer.wait_closed()

    async def process_request(self, msg: dict, client_id: str = "unknown") -> dict:
        """Process a single request and return a response."""
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})

        start_time = time.perf_counter()

        try:
            # Dispatch to agenda method dynamically
            if hasattr(self.agenda, method):
                agenda_method = getattr(self.agenda, method)
                result = await agenda_method(**params)
            else:
                raise ValueError(f"Unknown method: {method}")

            return {"id": req_id, "result": result}

        except Exception as e:
            # Distinguish expected conflicts from real errors
            if "already in status" in str(e):
                logger.debug(f"Task claim conflict (expected): {e}")
            else:
                logger.error(f"Error processing request {method}: {e}", exc_info=True)
            return {"id": req_id, "error": str(e)}

        finally:
            # Record call latency with client identifier
            if self._performance_tracker is not None and method:
                latency = time.perf_counter() - start_time
                self._performance_tracker.record_call(method, latency, client_id)

    async def start(self):
        """Start the server on TCP or Unix socket."""
        if self.unix_socket:
            # Clean up existing socket file if it exists
            if os.path.exists(self.unix_socket):
                os.unlink(self.unix_socket)
            self.server = await asyncio.start_unix_server(
                self.handle_client,
                self.unix_socket
            )
            logger.info(f"AgendaServer listening on Unix socket {self.unix_socket}")
        else:
            self.server = await asyncio.start_server(
                self.handle_client,
                self.host,
                self.port
            )
            addrs = ', '.join(str(sock.getsockname()) for sock in self.server.sockets)
            logger.info(f"AgendaServer listening on {addrs}")

            # Write server address to file if requested
            if self.server_address_path:
                hostname = socket.gethostname()
                self._write_server_address(hostname, self.port)

        self._running = True
        async with self.server:
            await self.server.serve_forever()

    def _write_server_address(self, host: str, port: int):
        """Write server address info to a JSON file for clients to discover."""
        address_info = {
            "host": host,
            "port": port
        }

        try:
            # Write atomically by writing to temp file and renaming
            tmp_path = f"{self.server_address_path}.tmp"
            with open(tmp_path, 'w') as f:
                json.dump(address_info, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.server_address_path)
            logger.info(f"Wrote server address to {self.server_address_path}: {address_info}")
        except Exception as e:
            logger.error(f"Failed to write server address to {self.server_address_path}: {e}")

    async def serve_pipe(self, read_fd: int, write_fd: int):
        """Serve a single client over a pipe (e.g., from os.pipe())."""
        loop = asyncio.get_event_loop()

        # Create stream reader/writer from file descriptors
        reader = asyncio.StreamReader()
        read_protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: read_protocol, os.fdopen(read_fd, 'rb'))

        write_transport, write_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin,
            os.fdopen(write_fd, 'wb')
        )
        writer = asyncio.StreamWriter(write_transport, write_protocol, reader, loop)

        logger.info(f"AgendaServer serving over pipe (read_fd={read_fd}, write_fd={write_fd})")
        await self.handle_client(reader, writer)

    async def stop(self):
        """Stop the server and checkpoint."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self._running = False
        self._sync_checkpoint()
        logger.info("AgendaServer stopped")


class AgendaClient(Agenda):
    """
    Client implementation of the Agenda protocol.

    Forwards all method calls to a remote AgendaServer over any stream transport.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        unix_socket: Optional[str] = None,
        reader: Optional[asyncio.StreamReader] = None,
        writer: Optional[asyncio.StreamWriter] = None,
        timeout: float = 30.0,
        server_address_path: Optional[str] = None,
    ):
        """
        Initialize AgendaClient.

        Provide either:
        - host and port for TCP connection
        - unix_socket for Unix domain socket
        - reader and writer for custom stream (e.g., pipe)
        - server_address_path to read host and port from a JSON file
        """
        self.host = host
        self.port = port
        self.unix_socket = unix_socket
        self.timeout = timeout
        self.server_address_path = server_address_path
        self._request_counter = 0
        self._reader = reader
        self._writer = writer
        self._lock = asyncio.Lock()
        self._owns_connection = (reader is None and writer is None)
        self._address_loaded = False

    @classmethod
    async def from_pipe(cls, read_fd: int, write_fd: int):
        """Create a client connected via pipe file descriptors."""
        loop = asyncio.get_event_loop()

        # Create stream reader/writer from file descriptors
        reader = asyncio.StreamReader()
        read_protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: read_protocol, os.fdopen(read_fd, 'rb'))

        write_transport, write_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin,
            os.fdopen(write_fd, 'wb')
        )
        writer = asyncio.StreamWriter(write_transport, write_protocol, reader, loop)

        return cls(reader=reader, writer=writer)

    def _load_server_address(self):
        """Load server address from JSON file if server_address_path is set."""
        if self.server_address_path and not self._address_loaded:
            try:
                with open(self.server_address_path, 'r') as f:
                    address_info = json.load(f)
                self.host = address_info.get("host")
                self.port = address_info.get("port")
                self._address_loaded = True
                logger.info(f"Loaded server address from {self.server_address_path}: {self.host}:{self.port}")
            except FileNotFoundError:
                raise RuntimeError(f"Server address file not found: {self.server_address_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load server address from {self.server_address_path}: {e}")

    async def _connect(self):
        """Ensure connection to server."""
        if self._writer is None or self._writer.is_closing():
            # Load server address from file if needed
            self._load_server_address()

            if self.unix_socket:
                self._reader, self._writer = await asyncio.open_unix_connection(self.unix_socket)
                logger.debug(f"Connected to AgendaServer at {self.unix_socket}")
            elif self.host and self.port:
                self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
                logger.debug(f"Connected to AgendaServer at {self.host}:{self.port}")
            else:
                raise RuntimeError("No connection parameters provided")

    async def _call(self, method: str, **params) -> Any:
        """Make a remote procedure call to the server."""
        async with self._lock:
            if self._owns_connection:
                await self._connect()

            # Build request
            self._request_counter += 1
            req_id = self._request_counter

            request = {
                "id": req_id,
                "method": method,
                "params": params,
            }

            # Send request and read response
            await _write_message(self._writer, request)
            response = await _read_message(self._reader)

            # Check for errors
            if "error" in response:
                raise RuntimeError(f"Remote error: {response['error']}")

            return response.get("result")

    async def get_object(self, path: str) -> Optional[Object]:
        return await self._call("get_object", path=path)

    async def create_object(self, obj: Object) -> str:
        return await self._call("create_object", obj=obj)

    async def update_object(
        self,
        path: str,
        new_content: Optional[bytes] = None,
        new_properties: Optional[dict[str, Any]] = None,
        interest_factor: Optional[float] = None,
        interest_recursion_gamma: Optional[float] = None,
    ) -> None:
        return await self._call(
            "update_object",
            path=path,
            new_content=new_content,
            new_properties=new_properties,
            interest_factor=interest_factor,
            interest_recursion_gamma=interest_recursion_gamma,
        )

    async def add_task(self, task: Task) -> str:
        return await self._call("add_task", task=task)

    async def get_tasks(
        self,
        type: Optional[str] = None,
        ignore_completed: bool = False,
    ) -> list[tuple[Task, TaskStatus]]:
        return await self._call(
            "get_tasks",
            type=type,
            ignore_completed=ignore_completed,
        )

    async def update_task(
        self,
        task_id: str,
        priority_factor: Optional[float] = None,
        recursion_gamma: Optional[float] = None,
        work_status: Optional[WorkStatus] = None,
        new_notes: Optional[dict[str, Any]] = None,
    ):
        return await self._call(
            "update_task",
            task_id=task_id,
            priority_factor=priority_factor,
            recursion_gamma=recursion_gamma,
            work_status=work_status,
            new_notes=new_notes,
        )

    async def claim_next_tasks(
        self,
        type: Optional[str] = None,
        batch_size: int = 1,
    ) -> Optional[list[tuple[Task, TaskStatus]]]:
        return await self._call("claim_next_tasks", type=type, batch_size=batch_size)

    async def close(self):
        """Close the connection to the server."""
        if self._writer and not self._writer.is_closing() and self._owns_connection:
            self._writer.close()
            await self._writer.wait_closed()
            logger.debug("Disconnected from AgendaServer")
