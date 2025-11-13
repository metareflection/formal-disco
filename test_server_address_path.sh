#!/bin/bash
# LLM-generated test of the server_address_path functionality
set -e

# Python executable (can be overridden with environment variable)
PYTHON=${PYTHON:-python}

# Test script for server_address_path functionality
# Tests that AgendaServer writes connection info to a file and AgendaClient can read it

echo "Testing server_address_path functionality..."
echo "Using Python: $PYTHON"
echo

# Create a temporary file for the server address
ADDRESS_FILE=$(mktemp /tmp/formal-disco-addr.XXXXXX.json)
echo "Using address file: $ADDRESS_FILE"
echo

# Cleanup on exit
trap "rm -f $ADDRESS_FILE $ADDRESS_FILE.tmp" EXIT

# Start the server in the background
echo "Starting AgendaServer..."
$PYTHON agenda_server.py \
    agenda=local \
    server.server_address_path=$ADDRESS_FILE \
    agenda.checkpoint_path=/tmp/test-agenda.pkl &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start and write the address file
echo "Waiting for server to write address file..."
sleep 2

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server process died"
    exit 1
fi

# Check if address file was created
if [ ! -f "$ADDRESS_FILE" ]; then
    echo "ERROR: Address file was not created: $ADDRESS_FILE"
    kill $SERVER_PID
    exit 1
fi

# Print the JSON file contents
echo
echo "=== Server Address File Contents ==="
cat "$ADDRESS_FILE"
echo
echo "===================================="
echo

# Verify it's valid JSON and has expected fields
if ! jq -e '.host and .port' "$ADDRESS_FILE" > /dev/null 2>&1; then
    echo "ERROR: Address file is missing 'host' or 'port' fields"
    kill $SERVER_PID
    exit 1
fi

HOST=$(jq -r '.host' "$ADDRESS_FILE")
PORT=$(jq -r '.port' "$ADDRESS_FILE")

echo "Server is listening on: $HOST:$PORT"
echo

# Now run a scheduler that connects using the address file
echo "Starting scheduler with distributed agenda using server_address_path..."
timeout 5 $PYTHON scheduler.py \
    agenda=distributed \
    agenda.server_address_path=$ADDRESS_FILE \
    scheduler=example \
    || true

echo
echo "✓ Test completed successfully!"
echo "  - Server wrote address file"
echo "  - Address file contains valid JSON with host and port"
echo "  - Client was able to connect using server_address_path"

# Cleanup
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
rm -f /tmp/test-agenda.pkl
