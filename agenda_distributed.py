#!/usr/bin/env python3

"""
Distributed Agenda implementation using client-server architecture.

AgendaServer wraps a LocalAgenda and serves requests over any stream transport.
AgendaClient implements the Agenda protocol by forwarding requests to the server.

Both support TCP connections, Unix domain sockets, and os.pipe().
"""

import asyncio
import atexit
import logging
import os
import pickle
import signal
import sys
from typing import Any, Optional, Union

from agenda import Agenda, LocalAgenda, Object, Task, TaskStatus, WorkStatus

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
    ):
        self.agenda = agenda
        self.host = host
        self.port = port
        self.unix_socket = unix_socket
        self.server: Optional[asyncio.Server] = None
        self._running = False
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
        logger.debug(f"Client connected from {addr}")

        try:
            while True:
                msg = await _read_message(reader)
                response = await self.process_request(msg)
                await _write_message(writer, response)

        except asyncio.IncompleteReadError:
            logger.debug(f"Client {addr} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}", exc_info=True)
        finally:
            writer.close()
            await writer.wait_closed()

    async def process_request(self, msg: dict) -> dict:
        """Process a single request and return a response."""
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})

        try:
            # Dispatch to agenda method
            if method == "get_object":
                result = await self.agenda.get_object(**params)
            elif method == "create_object":
                result = await self.agenda.create_object(**params)
            elif method == "update_object":
                result = await self.agenda.update_object(**params)
            elif method == "add_task":
                result = await self.agenda.add_task(**params)
            elif method == "get_tasks":
                result = await self.agenda.get_tasks(**params)
            elif method == "update_task":
                result = await self.agenda.update_task(**params)
            elif method == "update_priority":
                result = await self.agenda.update_priority(**params)
            elif method == "update_status":
                result = await self.agenda.update_status(**params)
            elif method == "update_notes":
                result = await self.agenda.update_notes(**params)
            elif method == "claim_next_task":
                result = await self.agenda.claim_next_task(**params)
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

        self._running = True
        async with self.server:
            await self.server.serve_forever()

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
    ):
        """
        Initialize AgendaClient.

        Provide either:
        - host and port for TCP connection
        - unix_socket for Unix domain socket
        - reader and writer for custom stream (e.g., pipe)
        """
        self.host = host
        self.port = port
        self.unix_socket = unix_socket
        self.timeout = timeout
        self._request_counter = 0
        self._reader = reader
        self._writer = writer
        self._lock = asyncio.Lock()
        self._owns_connection = (reader is None and writer is None)

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

    async def _connect(self):
        """Ensure connection to server."""
        if self._writer is None or self._writer.is_closing():
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
    ) -> None:
        return await self._call(
            "update_object",
            path=path,
            new_content=new_content,
            new_properties=new_properties,
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

    async def claim_next_task(
        self,
        type: Optional[str] = None,
    ) -> Optional[tuple[Task, TaskStatus]]:
        return await self._call("claim_next_task", type=type)

    async def close(self):
        """Close the connection to the server."""
        if self._writer and not self._writer.is_closing() and self._owns_connection:
            self._writer.close()
            await self._writer.wait_closed()
            logger.debug("Disconnected from AgendaServer")
