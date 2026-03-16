# Distributed Agenda

With the distributed agenda, workers run in separate processes (or even on different machines) while sharing a common agenda.

## Architecture

The distributed agenda uses a client-server architecture:

- **AgendaServer**: Wraps a `LocalAgenda` and serves requests over TCP, Unix sockets, or pipes
- **AgendaClient**: Implements the `Agenda` protocol by forwarding all calls to the server
- **Protocol**: Simple pickle-based RPC with length-prefixed messages

## Features

- **Transparent**: AgendaClient implements the same `Agenda` protocol as LocalAgenda
- **Checkpointing**: The server handles all checkpointing, both periodically and on termination signals (SIGTERM, SIGINT)
- **Flexible Transport**: Supports TCP, Unix domain sockets, and pipes
- **Concurrent Workers**: Multiple workers can safely operate on the same agenda

## Usage

### 1. Start the AgendaServer

```bash
# TCP server (default)
python agenda_server.py --host 127.0.0.1 --port 9999 --checkpoint agenda.pkl

# Unix socket server
python agenda_server.py --unix-socket /tmp/agenda.sock --checkpoint agenda.pkl
```

### 2. Run Workers

Start worker processes that connect to the server:

```bash
# Idea generator worker
python agenda_client.py --port 9999 --worker-type generator --fuel 10

# Implementer worker (in another terminal/process)
python agenda_client.py --port 9999 --worker-type implementer --fuel 10
```

### 3. Using Hydra Configuration

You can also use Hydra configs:

```bash
# Server
python agenda_server.py --hydra-config local

# Client
python agenda_client.py --hydra-config distributed_example
```

## Quick Tests

### Basic Distributed Test

Run the provided test script to see the distributed system in action:

```bash
bash test_distributed.sh
```

This script:
1. Starts an AgendaServer in the background
2. Runs DummyIdeaGenerator and DummyImplementer workers in separate processes
3. Both workers connect to the same server and work concurrently for 5 seconds
4. Checkpoints are saved periodically and on shutdown

### Multi-Client Competition Test

Test multiple workers competing for the same tasks:

```bash
# Run with 3 workers and 15 tasks
bash test_multi_client.sh 3 15

# Run with 6 workers and 30 tasks
bash test_multi_client.sh 6 30
```

This script:
1. Starts an AgendaServer in the background
2. Generates N tasks using DummyIdeaGenerator
3. Starts M implementer workers that compete for the tasks
4. Shows task distribution across workers
5. Verifies all tasks are completed without conflicts

After the test completes, materialize the results:

```bash
python materialize.py distributed-test-agenda.pkl output/
# or
python materialize.py multi-client-test-agenda.pkl output-multi/
```

## Configuration

### AgendaServer

```python
from agenda import LocalAgenda
from agenda_distributed import AgendaServer

local_agenda = LocalAgenda(
    checkpoint_path="agenda.pkl",
    checkpoint_interval=50,
)

server = AgendaServer(
    local_agenda,
    host="127.0.0.1",
    port=9999,
)

await server.start()
```

### AgendaClient

```python
from agenda_distributed import AgendaClient

# TCP connection
agenda = AgendaClient(host="127.0.0.1", port=9999)

# Unix socket connection
agenda = AgendaClient(unix_socket="/tmp/agenda.sock")

# Use like any other agenda
task_id = await agenda.add_task(task)
tasks = await agenda.get_tasks(type="implement")
```

## Using with Pipes

For parent-child process communication:

```python
import os
from agenda_distributed import AgendaServer, AgendaClient

# Create pipes
parent_read, child_write = os.pipe()
child_read, parent_write = os.pipe()

# In parent process
client = await AgendaClient.from_pipe(parent_read, parent_write)

# In child process
await server.serve_pipe(child_read, child_write)
```

## Signal Handling

The AgendaServer handles termination signals gracefully:
- **SIGTERM**: Checkpoint and exit cleanly
- **SIGINT**: Checkpoint and exit cleanly (Ctrl-C)
- **atexit**: Checkpoint on normal exit

This ensures that no work is lost when shutting down the server.

## Comparison with LocalAgenda

| Feature | LocalAgenda | Distributed Agenda |
|---------|-------------|-------------------|
| Workers in same process | ✓ | ✗ |
| Workers in different processes | ✗ | ✓ |
| Workers on different machines | ✗ | ✓ (TCP) |
| Checkpointing | Manual | Automatic (server) |
| Signal handling | Basic | Comprehensive |
| Complexity | Simple | Moderate |

## Protocol Details

The protocol uses length-prefixed pickle messages:

**Request:**
```python
{
    "id": int,           # Request ID
    "method": str,       # Method name (e.g., "add_task")
    "params": dict,      # Method parameters
}
```

**Response:**
```python
{
    "id": int,           # Matching request ID
    "result": Any,       # Return value
}
# or
{
    "id": int,           # Matching request ID
    "error": str,        # Error message
}
```

Messages are sent as: `[4-byte length][pickled message]`

## Next steps
Potential improvements:
- Async batching of operations on a single server?
- Explicit load balancing (do we need this?)
