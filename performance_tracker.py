#!/usr/bin/env python3

"""
Performance tracking for distributed agenda RPC calls.

Tracks call latency and computes rolling statistics over a configurable time window.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class CallRecord:
    """A single RPC call record."""
    procedure: str
    timestamp: float  # Unix timestamp when call completed
    latency: float    # Latency in seconds
    client: str       # Client identifier (e.g., IP:port)


class PerformanceTracker:
    """
    Tracks RPC call performance with rolling statistics.

    Maintains a deque of call records and computes statistics over a
    configurable time window (default 10 minutes).
    """

    def __init__(self, window_seconds: float = 600.0):
        """
        Initialize the performance tracker.

        Args:
            window_seconds: Time window for statistics (default 600s = 10 min)
        """
        self._window_seconds = window_seconds
        self._records: deque[CallRecord] = deque()

    def record_call(self, procedure: str, latency: float, client: str = "unknown") -> None:
        """
        Record a completed RPC call.

        Args:
            procedure: Name of the RPC method
            latency: Call latency in seconds
            client: Client identifier (e.g., IP:port or socket path)
        """
        now = time.time()
        self._records.append(CallRecord(
            procedure=procedure,
            timestamp=now,
            latency=latency,
            client=client,
        ))
        self._prune_old_records(now)

    def _prune_old_records(self, now: Optional[float] = None) -> None:
        """Remove records older than the time window."""
        if now is None:
            now = time.time()
        cutoff = now - self._window_seconds
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()

    def get_statistics(self) -> dict[str, dict[str, float]]:
        """
        Compute statistics for all procedures over the time window.

        Returns:
            Dict mapping procedure name to stats dict containing:
            - calls_per_min: Call rate (calls per minute)
            - avg_latency_ms: Average latency in milliseconds
            - max_latency_ms: Maximum latency in milliseconds
            - total_calls: Total number of calls in the window
        """
        self._prune_old_records()

        if not self._records:
            return {}

        # Group records by procedure
        by_procedure: dict[str, list[CallRecord]] = {}
        for record in self._records:
            if record.procedure not in by_procedure:
                by_procedure[record.procedure] = []
            by_procedure[record.procedure].append(record)

        # Compute time span for rate calculation
        now = time.time()
        oldest = self._records[0].timestamp
        time_span_minutes = max((now - oldest) / 60.0, 1/60.0)  # At least 1 second

        stats: dict[str, dict[str, float]] = {}
        for procedure, records in by_procedure.items():
            latencies = [r.latency for r in records]
            total_calls = len(records)

            stats[procedure] = {
                "calls_per_min": total_calls / time_span_minutes,
                "avg_latency_ms": (sum(latencies) / total_calls) * 1000,
                "max_latency_ms": max(latencies) * 1000,
                "total_calls": total_calls,
            }

        return stats

    def get_aggregate_statistics(self) -> dict[str, float]:
        """
        Compute aggregate statistics across all procedures.

        Returns:
            Dict containing:
            - total_calls_per_min: Total call rate across all procedures
            - overall_avg_latency_ms: Average latency across all calls
            - overall_max_latency_ms: Maximum latency across all calls
            - total_calls: Total number of calls
            - active_clients: Number of unique clients in the window
        """
        self._prune_old_records()

        if not self._records:
            return {
                "total_calls_per_min": 0.0,
                "overall_avg_latency_ms": 0.0,
                "overall_max_latency_ms": 0.0,
                "total_calls": 0,
                "active_clients": 0,
            }

        now = time.time()
        oldest = self._records[0].timestamp
        time_span_minutes = max((now - oldest) / 60.0, 1/60.0)

        latencies = [r.latency for r in self._records]
        total_calls = len(latencies)
        unique_clients = len(set(r.client for r in self._records))

        return {
            "total_calls_per_min": total_calls / time_span_minutes,
            "overall_avg_latency_ms": (sum(latencies) / total_calls) * 1000,
            "overall_max_latency_ms": max(latencies) * 1000,
            "total_calls": total_calls,
            "active_clients": unique_clients,
        }

    def get_client_statistics(self) -> dict[str, dict[str, float]]:
        """
        Compute statistics grouped by client.

        Returns:
            Dict mapping client identifier to stats dict containing:
            - calls_per_min: Call rate for this client
            - avg_latency_ms: Average latency in milliseconds
            - max_latency_ms: Maximum latency in milliseconds
            - total_calls: Total number of calls from this client
        """
        self._prune_old_records()

        if not self._records:
            return {}

        # Group records by client
        by_client: dict[str, list[CallRecord]] = {}
        for record in self._records:
            if record.client not in by_client:
                by_client[record.client] = []
            by_client[record.client].append(record)

        # Compute time span for rate calculation
        now = time.time()
        oldest = self._records[0].timestamp
        time_span_minutes = max((now - oldest) / 60.0, 1/60.0)

        stats: dict[str, dict[str, float]] = {}
        for client, records in by_client.items():
            latencies = [r.latency for r in records]
            total_calls = len(records)

            stats[client] = {
                "calls_per_min": total_calls / time_span_minutes,
                "avg_latency_ms": (sum(latencies) / total_calls) * 1000,
                "max_latency_ms": max(latencies) * 1000,
                "total_calls": total_calls,
            }

        return stats

    def get_active_clients(self) -> list[str]:
        """
        Get list of clients that have made calls within the time window.

        Returns:
            List of client identifiers sorted by most recent activity.
        """
        self._prune_old_records()

        if not self._records:
            return []

        # Get most recent timestamp per client
        last_seen: dict[str, float] = {}
        for record in self._records:
            if record.client not in last_seen or record.timestamp > last_seen[record.client]:
                last_seen[record.client] = record.timestamp

        # Sort by most recent first
        return sorted(last_seen.keys(), key=lambda c: last_seen[c], reverse=True)
