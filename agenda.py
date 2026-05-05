#!/usr/bin/env python3

"""
DiscoWorker agenda.

For now we only have a local implementation of the agenda, but the idea is that it will
be distributed so that we can spawn async workers on many machines.
"""

import asyncio
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import datetime
import math
import signal
import threading
import pickle
import os
import secrets
import uuid
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional, Iterable, Protocol

import numpy as np

from language import Language
from logger import AgendaLogger
from logger.noop import NoOpLogger
from patch import compute_reverse_patch
from performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)



@dataclass(slots=True, kw_only=True)
class Object:
    """
    Generic representation for the inputs and outputs of tasks.

      - path: string identifier. Corresponds to a path on disk.
      - type: arbitrary object "kind" (e.g., "dafny-program")
      - parents: list of parent object IDs (for lineage)
    - content: bytes with the actual content.
    - properties: arbitrary metadata bag
    - interestingness: float multiplier used to modulate priorities of tasks that depend on this object (default 1.0)
    """
    path: str
    type: str
    parents: list[str] = field(default_factory=list)
    content: Optional[bytes] = None
    properties: dict[str, Any] = field(default_factory=dict)
    interestingness: float = 1.0


@dataclass(slots=True, kw_only=True)
class Task:
    """
    A unit of work tracked by an Agenda.

      - id: string identifier (uuid)
      - type: arbitrary task "kind" (workers can filter tasks by this)
    - parents: list of parent task IDs (for lineage / propagation)
    - properties: arbitrary metadata bag
    - interest_dependencies: list of object paths whose interestingness multiplies this task's effective priority
    """
    id: str
    type: str
    parents: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    interest_dependencies: list[str] = field(default_factory=list)


class WorkStatus(StrEnum):
    NEW = "NEW"  # Newly created, never attempted task.
    ATTEMPTED = "ATTEMPTED"  # Not completed, but some worker has already tried this.
    DOING = "DOING"  # Some worker is currently working on this.
    DONE = "DONE"  # Completed.
    FAILED = "FAILED"  # Gave up on this.


    def signals_attempt_completed(self) -> bool:
        """Whether updating to this status indicates that a worker has completed an attempt at this task."""
        return self in (WorkStatus.ATTEMPTED, WorkStatus.DONE, WorkStatus.FAILED)


@dataclass(slots=True, kw_only=True)
class TaskStatus:
    """
    Mutable runtime status for a Task.
    """
    priority: float = 1.0
    work_status: WorkStatus = WorkStatus.NEW
    attempts: int = 0
    worker_notes: dict[str, Any] = field(default_factory=dict)

    def is_completed(self) -> bool:
        return self.work_status in (WorkStatus.DONE, WorkStatus.FAILED)


class StopWork(Exception):
    """Raised when workers should stop because the experiment is finished."""


class Agenda(Protocol):
    """
    Abstract async agenda. Keeps track of tasks and a database of objects.
    """

    async def get_object(self, path: str) -> Optional[Object]:
        """
        Return the object at `path`, or None if not found.
        """
        raise NotImplementedError

    async def create_object(self, obj: Object) -> str:
        """
        Create a new object. Returns the final path.

        The path can be modified in order to be made unique (e.g., by appending a uuid).
        """
        raise NotImplementedError

    async def update_object(self,
                            path: str,
                            new_content: Optional[bytes] = None,
                            new_properties: Optional[dict[str, Any]] = None,
                            interest_factor: Optional[float] = None,
                            interest_recursion_gamma: Optional[float] = None) -> None:
        """
        Update an existing object.

        Passing None for a field keeps it as is. Properties are merged if provided.
        """
        raise NotImplementedError

    async def update_task(
        self,
        task_id: str,
        priority_factor: Optional[float] = None,
        recursion_gamma: Optional[float] = None,
        work_status: Optional[WorkStatus] = None,
        new_notes: Optional[dict[str, Any]] = None,
    ):
        """
        Updates an existing task. Passing None for a field keeps it as is.

        This should fail if updating the status to DOING and it is already 'DOING'
        (meaning the task has been claimed).
        """
        raise NotImplementedError

    async def add_task(self, task: Task) -> str:
        """
        Register a new task and return an auto-assigned id.
        """
        raise NotImplementedError

    async def get_tasks(
        self,
        type: Optional[str] = None,
        ignore_completed: bool = False,
    ) -> list[tuple[Task, TaskStatus]]:
        """
        Return (Task, TaskStatus) pairs sorted by descending effective priority.
        If `type` is provided, filter by exact match.
        If `ignore_completed` is True, exclude DONE/FAILED.
        """
        raise NotImplementedError

    async def update_priority(
        self,
        task_id: str,
        priority_factor: float,
        recursion_gamma: float = 0.0,
    ) -> None:
        await self.update_task(
            task_id,
            priority_factor=priority_factor,
            recursion_gamma=recursion_gamma,
        )

    async def update_status(
        self,
        task_id: str,
        work_status: WorkStatus,
    ) -> None:
        await self.update_task(task_id, work_status=work_status)

    async def update_notes(
        self,
        task_id: str,
        new_notes: dict[str, Any],
    ) -> None:
        await self.update_task(task_id, new_notes=new_notes)

    async def claim_next_tasks(
        self,
        type: Optional[str] = None,
        batch_size: int = 1,
    ) -> Optional[list[tuple[Task, TaskStatus]]]:
        """
        Atomically find and claim the next available task(s).

        Args:
            type: Optional task type filter
            batch_size: Number of tasks to claim (default 1)

        Returns:
            List of (Task, TaskStatus) tuples if tasks were claimed, None if no tasks available.
            For batch_size=1, returns a single-element list or None.
            This is more efficient than separate get_tasks() + update_status() calls.
        """
        raise NotImplementedError


class LocalAgenda(Agenda):
    """
    In-memory agenda using dict/list structures and an asyncio.Lock
    for atomic multi-field updates.

    Periodically checkpoints to disk if checkpoint_path is provided.

    If max_attempts is provided, the agenda will raise StopWork on most
    methods to signal that workers should stop. This is used to bound
    experiments by number of task attempts and allow us to try to fairly
    make comparisons between different runs of the full system.

    Notes on priority propagation (UPWARDS):
      - Setting recursion_gamma <= 0 disables propagation.
      - With recursion_gamma in (0, 1], we propagate to ancestors:
          priority(ancestor at distance d) := max(current, root_priority * gamma**d)
        where distance d >= 1 (parent: d=1, grandparent: d=2, ...).
      - We discover ancestors via the `parents` lists stored on each Task.
        If an ancestor task is not present in this agenda, that branch stops.
    """

    def __init__(
            self,
            checkpoint_path: Optional[str] = None,
            checkpoint_interval: Optional[int] = 50,
            logger: Optional[AgendaLogger] = None,
            benchmark_codebase_time: Optional[int] = 5*60,
            performance_tracker: Optional[PerformanceTracker] = None,
            sort_every: int = 100,
            language: str = 'dafny',
            max_attempts: Optional[int] = None,
    ) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, Task] = {}
        self._status: dict[str, TaskStatus] = {}
        self._objects: dict[str, Object] = {}
        self._language = language.lower()
        self._backend = Language[language.upper()].get_backend()
        self._total_attempts = 0
        self._max_attempts = max_attempts

        self._clock = 0
        self._checkpoint_path = checkpoint_path
        self._checkpoint_interval = checkpoint_interval
        self._signal_ckpt_once = threading.Event()
        self._logger = logger or NoOpLogger()
        self._performance_tracker = performance_tracker

        self._sort_every = sort_every
        self._sorted_task_ids: list[str] = []
        self._sort_calls: int = 0

        # Cumulative outcome counts: task_type -> {WorkStatus value -> count}.
        # Persisted in checkpoints so rates accumulate across restarts.
        self._task_outcomes: dict[str, dict[str, int]] = {}

        self._patch_executor = ProcessPoolExecutor(max_workers=1)

        self._load()

        self._benchmark_codebase_at = \
            (None if benchmark_codebase_time is None
             else datetime.datetime.now() +
                  datetime.timedelta(seconds=benchmark_codebase_time))

        # We used to checkpoint at atexit and register that here, but not
        # anymore: now, if the run is killed before it finishes, we will
        # just keep the last checkpoint and lose a bit of work. It is tricky
        # to checkpoint on a signal because checkpointing takes time when
        # the agenda is large, and usually times out and potentially leaves
        # corrupt files on disk.


    def _load(self):
        if self._checkpoint_path is None:
            return
        try:
            with open(self._checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                self._tasks = data['tasks']
                self._status = data['status']
                self._total_attempts = data.get('total_attempts', 0)

                # Reset all task statuses to 'ATTEMPTED' if they were 'DOING':
                reset = 0
                for k, v in self._status.items():
                    if v.work_status == WorkStatus.DOING:
                        v.work_status = WorkStatus.ATTEMPTED
                        reset += 1

                logger.info(f'{reset} tasks had DOING status; reset to ATTEMPTED')

                self._objects = data.get('objects', {})
                self._clock = data.get('clock', 0)
                self._task_outcomes = data.get('task_outcomes', {})
            # Build initial sorted task ID cache.
            self._rebuild_sorted_task_ids()
            stats = self._compute_codebase_statistics()
            logger.info(f"Loaded agenda checkpoint from {self._checkpoint_path}.")
            logger.info(f"Codebase statistics: {stats}")
        except FileNotFoundError:
            logger.info('No checkpoint; starting empty agenda.')
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    def _should_stop(self) -> bool:
        return self._max_attempts is not None and self._total_attempts >= self._max_attempts

    async def _checkpoint(self):
        if self._checkpoint_path is None:
            return

        # Add a random suffix so concurrent checkpoints don't interfere on
        # each other's temp file (the rename is still atomic; last writer wins).
        # We don't really have concurrent checkpoints anymore, but this is still here
        # for precaution if we change again in the future.
        tmp_path = f"{self._checkpoint_path}.{os.getpid()}.{secrets.token_hex(4)}.new"
        async with self._lock:
            try:
                with open(tmp_path, 'wb') as f:
                    data = {
                        'tasks': self._tasks,
                        'status': self._status,
                        'clock': self._clock,
                        'objects': self._objects,
                        'task_outcomes': self._task_outcomes,
                        'total_attempts': self._total_attempts,
                    }
                    pickle.dump(data, f)
                    f.flush()
                    os.fsync(f.fileno())
                # Atomic rename: overwrite target with new file
                os.replace(tmp_path, self._checkpoint_path)
                logger.info(f"Checkpointed agenda to {self._checkpoint_path}.")

                # Run global stats (slower) logging every time we checkpoint.
                s = self._compute_codebase_statistics()
                self._logger.log_code_base_statistics(s)

                progress_metrics = {'tasks/total_attempts': self._total_attempts}
                if self._max_attempts is not None:
                    progress_metrics['tasks/max_attempts'] = self._max_attempts
                    progress_metrics['tasks/progress'] = self._total_attempts / self._max_attempts
                self._logger.log_metrics(progress_metrics)
                logger.info(f"Codebase statistics: {s}")

                d = self._compute_diversity_metrics()
                self._logger.log_metrics(d)
                logger.info(f"Diversity/complexity metrics: {d}")

                self._logger.log_task_outcomes(self._task_outcomes)
                logger.info(f"Task outcomes: {self._task_outcomes}")

                # Log performance statistics if tracker is available
                if self._performance_tracker is not None:
                    per_proc_stats = self._performance_tracker.get_statistics()
                    agg_stats = self._performance_tracker.get_aggregate_statistics()
                    self._logger.log_performance_statistics(per_proc_stats, agg_stats)
                    logger.info(f"RPC performance (aggregate): {agg_stats}")

            except Exception as e:
                logger.warning(f"Failed to write checkpoint: {e}")
                # Clean up tmp file if something went wrong
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass


    def tick(self):
        """
        Increment clock and checkpoint if needed.
        """
        self._clock += 1

        if self._benchmark_codebase_at:
            if datetime.datetime.now() >= self._benchmark_codebase_at:
                s = self._compute_codebase_statistics()
                s = {f'benchmark/{k}': v for k, v in s.items()}
                self._logger.log_code_base_statistics(s)
                self._benchmark_codebase_at = None

        if self._checkpoint_interval and self._clock % self._checkpoint_interval == 0:
            self._checkpoint()

        if self._should_stop():
            raise StopWork("Max task attempts reached in the agenda")


    def _all_ancestors(self, root_id: str) -> Iterable[tuple[str, int]]:
        """
        BFS over parents to enumerate (task_id, distance) for all ancestors.
        Distance(root) is 0; we yield only ancestors (distance >= 1).

        If a referenced parent isn't present in the agenda, we stop traversing.
        """
        queue: list[tuple[str, int]] = [(root_id, 0)]
        seen = {root_id}
        while queue:
            cur, d = queue.pop(0)
            task = self._tasks.get(cur)
            if task is None:
                continue
            for parent_id in task.parents:
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                queue.append((parent_id, d + 1))
                yield (parent_id, d + 1)

    def _all_object_ancestors(self, root_path: str) -> Iterable[tuple[str, int]]:
        """
        BFS over object parents to enumerate (object_path, distance) for all ancestors.
        Distance(root) is 0; we yield only ancestors (distance >= 1).

        If a referenced parent isn't present in the agenda, we stop traversing.
        """
        queue: list[tuple[str, int]] = [(root_path, 0)]
        seen = {root_path}
        while queue:
            cur, d = queue.pop(0)
            obj = self._objects.get(cur)
            if obj is None:
                continue
            for parent_path in obj.parents:
                if parent_path in seen:
                    continue
                seen.add(parent_path)
                queue.append((parent_path, d + 1))
                yield (parent_path, d + 1)

    async def add_task(self, task: Task) -> str:
        self.tick()

        async with self._lock:
            task.id = task.id or ''

            # Ensure ID is unique.
            if not task.id or task.id in self._tasks:
                task.id += str(uuid.uuid4())

            self._tasks[task.id] = task
            self._status[task.id] = TaskStatus()

            # Append to cached ordering (O(1)); full re-sort happens periodically.
            self._sorted_task_ids.append(task.id)

            # This task is created in the NEW state.
            self._logger.log_task_state(task.type, task.id, WorkStatus.NEW.value)

            return task.id

    def _effective_priority(self, task: Task, status: TaskStatus) -> float:
        """
        Compute effective priority: base priority multiplied by the product
        of interestingness of all dependency objects.

        This method is NOT thread-safe and should only be called while holding self._lock.
        """
        eff = status.priority
        for obj_path in getattr(task, 'interest_dependencies', []) or []:
            obj = self._objects.get(obj_path)
            if obj is None:
                continue
            try:
                eff *= float(getattr(obj, 'interestingness', 1.0))
            except Exception:
                pass
        return eff

    def _rebuild_sorted_task_ids(self) -> None:
        """
        Full sort of all task IDs by effective priority (descending).
        NOT thread-safe — call only while holding self._lock or during init.
        """
        pairs = []
        for tid, t in self._tasks.items():
            s = self._status[tid]
            pairs.append((tid, self._effective_priority(t, s)))
        pairs.sort(key=lambda p: p[1], reverse=True)
        self._sorted_task_ids = [tid for tid, _ in pairs]
        self._sort_calls = 0

    def _get_sorted_tasks(
        self,
        type: Optional[str] = None,
        ignore_completed: bool = False,
    ) -> Iterable[tuple[Task, TaskStatus]]:
        """
        Yield tasks in cached priority order (highest first).

        The cached ordering is rebuilt every ``sort_every`` calls.
        Callers that only need the first match can stop early.

        This method is NOT thread-safe and should only be called while holding self._lock.
        """
        self._sort_calls += 1
        if self._sort_calls >= self._sort_every:
            self._rebuild_sorted_task_ids()

        for tid in self._sorted_task_ids:
            t = self._tasks.get(tid)
            if t is None:
                continue
            if type is not None and t.type != type:
                continue
            s = self._status[tid]
            if ignore_completed and s.is_completed():
                continue
            yield (t, s)

    async def get_tasks(
        self,
        type: Optional[str] = None,
        ignore_completed: bool = False,
    ) -> list[tuple[Task, TaskStatus]]:
        async with self._lock:
            return list(self._get_sorted_tasks(type=type, ignore_completed=ignore_completed))

    async def update_task(
        self,
        task_id: str,
        priority_factor: Optional[float] = None,
        recursion_gamma: Optional[float] = None,
        work_status: Optional[WorkStatus] = None,
        new_notes: Optional[dict[str, Any]] = None,
    ):
        """
        Atomic update of a task's status fields, with optional recursive
        multiplicative priority update (to this task and ancestors).

        Priority is always updated by multiplying the existing priority
        by `priority_factor`, not resetting it. The intuition is that a task
        may increase or decrease in importance depending on what workers do.

        If a worker wants to determine that a task should not be pursued anymore,
        it can set work_status to DONE or FAILED. Otherwise, the task should be
        eventually picked up based on its priority.

        Behavior:
          - If priority is not None: set the root task's priority.
          - If recursion_gamma is not None and > 0: propagate priority update
            to ancestors, multiplicatively: multiply each ancestor's priority
            by priority * gamma**distance
          - If work_status is not None: set it, and bump attempts when moving
            into ATTEMPTED, DONE, or FAILED.
          - If new_notes is provided: do a shallow merge into current worker_notes.
        """

        self.tick()

        async with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Unknown task_id: {task_id}")

            status = self._status[task_id]

            if work_status == WorkStatus.DOING and status.work_status == WorkStatus.DOING:
                raise RuntimeError(f"Task {task_id} is already in status `DOING`.")

            if priority_factor is not None:
                status.priority *= priority_factor

                # Propagate priority update.
                if recursion_gamma is not None and recursion_gamma > 0:
                    for anc_id, dist in self._all_ancestors(task_id):
                        ds = self._status.get(anc_id)
                        if ds is None:
                            continue
                        propagated = priority_factor * (recursion_gamma ** dist)
                        if propagated > ds.priority:
                            ds.priority = propagated

            if work_status is not None:
                if work_status in (WorkStatus.ATTEMPTED, WorkStatus.DOING):
                    status.attempts += 1

                if work_status.signals_attempt_completed():
                    self._total_attempts += 1

                    # If we've hit the max attempts, signal workers to stop.
                    if self._should_stop():
                        logger.info(f"Max attempts {self._max_attempts} reached; signaling workers to stop.")
                        self._checkpoint()

                # Log status changes
                task = self._tasks[task_id]

                # Update status and inform logger of the new state so it can track current counts
                status.work_status = work_status
                self._logger.log_task_state(task.type, task_id, work_status.value)

                # Accumulate outcome counts for success-rate tracking.
                if work_status in (WorkStatus.DONE, WorkStatus.FAILED, WorkStatus.ATTEMPTED):
                    counts = self._task_outcomes.setdefault(task.type, {})
                    counts[work_status.value] = counts.get(work_status.value, 0) + 1

            if new_notes:
                status.worker_notes.update(new_notes)

    async def get_object(self, path: str) -> Optional[Object]:
        self.tick()
        async with self._lock:
            return self._objects.get(path)

    async def create_object(self, obj: Object) -> str:
        """
        Store a new object. If path already exists, add '_{r}' before the file
        extension (or at the end if there's no extension), where r is a random
        10-hex-character suffix. Keep trying until unique.
        """
        self.tick()
        async with self._lock:
            # If no path provided, start from a UUID.
            path = obj.path or str(uuid.uuid4())

            def split_root_ext(p: str) -> tuple[str, str]:
                # Split into root + ext, where ext includes the dot (".ext"), but only
                # if the dot appears after the last '/'. Otherwise, ext = "".
                slash_i = p.rfind("/")
                dot_i = p.rfind(".")
                if dot_i > slash_i:
                    return p[:dot_i], p[dot_i:]
                return p, ""

            # If collision, append "_{r}" before extension (or at end if no ext)
            while path in self._objects:
                r = uuid.uuid4().hex[:10]
                root, ext = split_root_ext(path)
                path = f"{root}_{r}{ext}"

            obj.path = path
            self._objects[path] = obj
            return path

    async def update_object(
        self,
        path: str,
        new_content: Optional[bytes] = None,
        new_properties: Optional[dict[str, Any]] = None,
        interest_factor: Optional[float] = None,
        interest_recursion_gamma: Optional[float] = None,
    ):
        """
        Update an existing object: replace content if provided, merge properties if provided.
        If interest_factor is provided, multiply the object's interestingness by this factor.
        If interest_recursion_gamma > 0, propagate a decayed multiplicative update to ancestors
        at distance d as interest_factor * (gamma ** d).
        """
        self.tick()

        old_bytes: Optional[bytes] = None

        # Phase 1 (under lock): swap content, update properties/interestingness.
        async with self._lock:
            if path not in self._objects:
                raise KeyError(f"Unknown object path: {path}")
            obj = self._objects[path]
            if new_content is not None:
                old_bytes = obj.content if obj.content is not None else b""
                obj.content = new_content
            if new_properties:
                obj.properties.update(new_properties)
            if interest_factor is not None:
                try:
                    obj.interestingness *= float(interest_factor)
                except Exception:
                    pass
                # Propagate to ancestors if requested
                if interest_recursion_gamma is not None and interest_recursion_gamma > 0:
                    for anc_path, dist in self._all_object_ancestors(path):
                        anc = self._objects.get(anc_path)
                        if anc is None:
                            continue
                        try:
                            propagated = float(interest_factor) * (float(interest_recursion_gamma) ** dist)
                            anc.interestingness *= propagated
                        except Exception:
                            pass

        # Phase 2 (outside lock): compute expensive reverse patch in a
        # separate process so the event loop can service other coroutines.
        if old_bytes is not None:
            loop = asyncio.get_running_loop()
            rev_patch = await loop.run_in_executor(
                self._patch_executor, compute_reverse_patch, new_content, old_bytes)

            # Phase 3 (under lock): append patch to history.
            async with self._lock:
                obj = self._objects[path]
                history = obj.properties.get("patch_history")
                if not isinstance(history, list):
                    history = []
                    obj.properties["patch_history"] = history
                history.append(rev_patch)

    def _compute_codebase_statistics(self) -> dict[str, int]:
        """
        Compute global statistics about the collection of programs in the dataset/ folder.
        For each parent_idea, only the longest program (by lines of code) is counted.
        """
        stats: dict[str, int] = {}

        # Collect all dataset programs grouped by parent_idea, keeping only the longest
        programs_by_idea: dict[str, tuple[Object, str, int]] = {}  # idea -> (obj, text, num_lines)

        for obj in self._objects.values():
            if obj.type != f"{self._language}-program" or not obj.path.startswith("dataset/"):
                continue
            content = obj.content
            if content is None:
                continue
            try:
                text = content.decode("utf-8")
            except Exception:
                continue

            lines = text.splitlines()
            num_lines = len(lines)
            parent_idea = obj.properties.get("parent_idea")

            # Keep only the longest program for each idea
            if parent_idea not in programs_by_idea or num_lines > programs_by_idea[parent_idea][2]:
                programs_by_idea[parent_idea] = (obj, text, num_lines)

        declaration_keywords = self._backend.declaration_keywords

        # Compute statistics on the filtered programs (one per idea)
        for parent_idea, (obj, text, num_lines) in programs_by_idea.items():
            lines = text.splitlines()

            stats["total_programs"] = stats.get("total_programs", 0) + 1
            stats["total_lines"] = stats.get("total_lines", 0) + num_lines

            verification_status = obj.properties.get("verification_status", "")

            if verification_status == "success":
                stats["total_verified_programs"] = stats.get("total_verified_programs", 0) + 1
                stats["loc_verified_programs"] = stats.get("loc_verified_programs", 0) + num_lines
            elif verification_status == "goal_unproven":
                stats["total_unproven_programs"] = stats.get("total_unproven_programs", 0) + 1
                stats["loc_unproven_programs"] = stats.get("loc_unproven_programs", 0) + num_lines

            for special in declaration_keywords:
                num_special = sum(1 for line in lines if line.strip().startswith(f"{special} "))
                # Always count towards _all (verified + unproven)
                stats[f"{special}_count_all"] = stats.get(f"{special}_count_all", 0) + num_special
                # Additionally count towards _verified for successful programs
                if verification_status == "success":
                    stats[f"{special}_count_verified"] = stats.get(f"{special}_count_verified", 0) + num_special

        return stats

    def _compute_diversity_metrics(self) -> dict[str, float]:
        """Compute diversity and complexity metrics over dataset programs.

        For each feature metric, computes the entropy of the pooled distribution
        across all programs (one per parent idea, taking the longest).
        For each complexity metric, computes the median and 90th percentile of
        all values collected across all programs.
        """
        from language import Program, Language as Lang

        # Same "longest per idea from dataset/" filter as _compute_codebase_statistics.
        programs_by_idea: dict[str, tuple[str, str, int]] = {}  # idea -> (path, text, n_lines)
        for obj in self._objects.values():
            if obj.type != f"{self._language}-program" or not obj.path.startswith("dataset/"):
                continue
            if obj.content is None:
                continue
            try:
                text = obj.content.decode("utf-8")
            except Exception:
                continue
            parent_idea = obj.properties.get("parent_idea")
            n_lines = len(text.splitlines())
            if parent_idea not in programs_by_idea or n_lines > programs_by_idea[parent_idea][2]:
                programs_by_idea[parent_idea] = (obj.path, text, n_lines)

        feature_totals: dict[str, Counter] = {}
        complexity_values: dict[str, list] = {}

        for path, text, _ in programs_by_idea.values():
            prog = Program(text, Lang[self._language.upper()], name=path)
            try:
                for metric, counter in self._backend.feature_sets(prog).items():
                    if metric not in feature_totals:
                        feature_totals[metric] = Counter()
                    feature_totals[metric] += counter
            except Exception:
                pass
            try:
                for metric, values in self._backend.complexity(prog).items():
                    if isinstance(values, list):
                        if metric not in complexity_values:
                            complexity_values[metric] = []
                        complexity_values[metric].extend(values)
            except Exception:
                pass

        stats: dict[str, float] = {}

        for metric, counter in feature_totals.items():
            total = sum(counter.values())
            if total == 0:
                continue
            entropy = -sum(
                (c / total) * math.log2(c / total)
                for c in counter.values() if c > 0
            )
            stats[f"diversity/{metric}-entropy"] = entropy

        for metric, values in complexity_values.items():
            if not values:
                continue
            arr = np.array(values, dtype=float)
            stats[f"complexity/{metric}-median"] = float(np.median(arr))
            stats[f"complexity/{metric}-p90"] = float(np.percentile(arr, 90))

        return stats

    async def claim_next_tasks(
        self,
        type: Optional[str] = None,
        batch_size: int = 1,
    ) -> Optional[list[tuple[Task, TaskStatus]]]:
        """
        Atomically find and claim the next available task(s) in a single lock acquisition.

        More efficient than get_tasks() + update_status() for high concurrency.
        Supports batch claiming for efficient batch inference with local LLMs.
        """
        self.tick()
        async with self._lock:
            # Get sorted tasks, excluding completed ones and only non-DOING tasks
            sorted_tasks = self._get_sorted_tasks(type=type, ignore_completed=True)

            # Claim up to batch_size tasks that are not currently being worked on
            claimed = []
            for task, status in sorted_tasks:
                if status.work_status != WorkStatus.DOING:
                    # Claim it
                    status.attempts += 1
                    status.work_status = WorkStatus.DOING
                    self._logger.log_task_state(task.type, task.id, WorkStatus.DOING.value)
                    claimed.append((task, status))

                    if len(claimed) >= batch_size:
                        break

            return claimed if claimed else None
