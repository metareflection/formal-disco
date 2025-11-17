#!/bin/bash
# Test script for multi-client distributed agenda
#
# This script demonstrates how multiple workers can connect to the same
# AgendaServer and compete for tasks without conflicts.
#
# Usage:
#   ./test_multi_client.sh [num_workers] [num_tasks]

set -e

# Use virtual environment python
shopt -s expand_aliases
alias python=venv/bin/python

# Configuration
NUM_WORKERS="${1:-1}"
NUM_TASKS="${2:-100}"
CHECKPOINT="multi-client-test-agenda.pkl"
PORT=9998
DURATION=120

echo "=== Multi-Client Distributed Agenda Test ==="
echo "Workers: $NUM_WORKERS"
echo "Tasks: $NUM_TASKS"
echo ""

# Clean up old checkpoint
rm -f "$CHECKPOINT"

# Start the server in the background
echo "1. Starting AgendaServer on port $PORT..."
python agenda_server.py --port $PORT --checkpoint "$CHECKPOINT" &
SERVER_PID=$!

# Give server time to start
sleep 2

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo "=== Cleaning up ==="

    # Kill all worker processes
    for pid in "${WORKER_PIDS[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            kill $pid 2>/dev/null || true
        fi
    done

    # Terminate server
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "Terminating server (PID $SERVER_PID)..."
        kill -TERM $SERVER_PID
        wait $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Generate initial tasks
echo "2. Generating $NUM_TASKS tasks..."
python agenda_client.py --port $PORT --worker-type generator --fuel $NUM_TASKS --duration 0.5 2>&1 | grep -E "(Generator|tasks)" || true

echo ""
echo "3. Starting $NUM_WORKERS implementer workers..."
echo ""

# Start multiple worker processes
WORKER_PIDS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "   Starting worker $i (PID will be assigned)..."
    python agenda_client.py --port $PORT --worker-type implementer --fuel 1 --duration $DURATION 2>&1 | \
        grep -E "(Worker|Claimed|completed|finished)" | \
        sed "s/Worker/Worker $i:/" &
    WORKER_PIDS+=($!)
    sleep 0.1  # Stagger start slightly
done

echo ""
echo "Workers running for up to ${DURATION} seconds..."
echo "  PIDs: ${WORKER_PIDS[*]}"
echo ""
echo "=== Worker Activity ==="
echo ""

# Wait for all workers to finish
for pid in "${WORKER_PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done

echo ""
echo "=== Test Results ==="
echo ""

# Analyze the checkpoint
python -c "
import pickle
import sys

with open('$CHECKPOINT', 'rb') as f:
    data = pickle.load(f)
    tasks = data['tasks']
    status = data['status']
    objects = data['objects']

    # Count task statuses
    from collections import Counter
    status_counts = Counter(s.work_status.value for s in status.values())

    # Count tasks by worker
    worker_tasks = {}
    for tid, s in status.items():
        worker_id = s.worker_notes.get('worker_id', 'unknown')
        if worker_id not in worker_tasks:
            worker_tasks[worker_id] = 0
        if s.work_status.value == 'DONE':
            worker_tasks[worker_id] += 1

    # Print summary
    print(f'Total tasks: {len(tasks)}')
    print(f'Task statuses:')
    for st, count in sorted(status_counts.items()):
        print(f'  {st}: {count}')
    print()
    print(f'Tasks completed by worker:')
    for wid in sorted(worker_tasks.keys(), key=lambda x: (x != 'unknown', x)):
        if worker_tasks[wid] > 0:
            print(f'  Worker {wid}: {worker_tasks[wid]} tasks')
    print()
    print(f'Total objects created: {len(objects)}')
    print(f'  Ideas: {sum(1 for o in objects.values() if o.type == \"idea\")}')
    print(f'  Programs: {sum(1 for o in objects.values() if o.type == \"dafny-program\")}')
" 2>/dev/null || echo "Could not analyze checkpoint (checkpoint may not exist)"

echo ""
echo "=== Checkpoint Details ==="
echo "Checkpoint file: $CHECKPOINT"
if [ -f "$CHECKPOINT" ]; then
    echo "Size: $(du -h $CHECKPOINT | cut -f1)"
fi

echo ""
echo "To view materialized results, run:"
echo "  python materialize.py $CHECKPOINT output-multi/"
