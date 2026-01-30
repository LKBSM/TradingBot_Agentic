# =============================================================================
# ASYNC HELPERS - Utilities for Asynchronous Operations
# =============================================================================
# Thread-safe queues and worker pools for non-blocking operations.
#
# =============================================================================

import queue
import threading
import logging
from typing import Optional, Callable, Any, List, Dict
from dataclasses import dataclass
from concurrent.futures import Future
import time


# =============================================================================
# ASYNC QUEUE
# =============================================================================

class AsyncQueue:
    """
    Thread-safe queue with worker processing.

    Example:
        q = AsyncQueue(processor=lambda x: print(f"Processing: {x}"))
        q.start()
        q.put("item1")
        q.put("item2")
        q.stop()
    """

    def __init__(
        self,
        processor: Callable[[Any], Any],
        max_size: int = 10000,
        num_workers: int = 1,
        name: str = "async-queue"
    ):
        self.processor = processor
        self.max_size = max_size
        self.num_workers = num_workers
        self.name = name
        self._logger = logging.getLogger(f"utils.{name}")

        self._queue = queue.Queue(maxsize=max_size)
        self._workers: List[threading.Thread] = []
        self._running = False

        self._stats = {
            'items_processed': 0,
            'items_failed': 0,
            'items_dropped': 0,
        }

    def start(self) -> None:
        """Start worker threads."""
        if self._running:
            return

        self._running = True
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"{self.name}-worker-{i}"
            )
            worker.start()
            self._workers.append(worker)

        self._logger.info(f"Started {self.num_workers} workers")

    def stop(self, wait: bool = True, timeout: float = 10.0) -> None:
        """Stop worker threads."""
        self._running = False

        # Put stop signals
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

        if wait:
            deadline = time.time() + timeout
            for worker in self._workers:
                remaining = deadline - time.time()
                if remaining > 0:
                    worker.join(timeout=remaining)

        self._workers.clear()

    def put(self, item: Any, block: bool = False) -> bool:
        """
        Put item in queue.

        Args:
            item: Item to process
            block: Wait if queue is full

        Returns:
            True if queued, False if dropped
        """
        try:
            if block:
                self._queue.put(item)
            else:
                self._queue.put_nowait(item)
            return True
        except queue.Full:
            self._stats['items_dropped'] += 1
            return False

    def _worker_loop(self) -> None:
        """Worker loop."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    break

                try:
                    self.processor(item)
                    self._stats['items_processed'] += 1
                except Exception as e:
                    self._logger.error(f"Processing error: {e}")
                    self._stats['items_failed'] += 1

            except queue.Empty:
                continue

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            'queue_size': self._queue.qsize(),
            'workers': len(self._workers),
            'running': self._running,
        }


# =============================================================================
# ASYNC WORKER POOL
# =============================================================================

@dataclass
class WorkItem:
    """Work item for async processing."""
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    future: Future = None
    priority: int = 0

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.future is None:
            self.future = Future()

    def __lt__(self, other):
        return self.priority < other.priority


class AsyncWorkerPool:
    """
    Generic async worker pool for any callable.

    Example:
        pool = AsyncWorkerPool(num_workers=4)
        pool.start()

        # Submit work
        future = pool.submit(expensive_function, arg1, arg2)
        result = future.result(timeout=5.0)

        pool.stop()
    """

    def __init__(
        self,
        num_workers: int = 4,
        max_queue_size: int = 1000,
        name: str = "worker-pool"
    ):
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.name = name
        self._logger = logging.getLogger(f"utils.{name}")

        self._queue = queue.PriorityQueue(maxsize=max_queue_size)
        self._workers: List[threading.Thread] = []
        self._running = False
        self._counter = 0  # For FIFO ordering within same priority
        self._lock = threading.Lock()

        self._stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
        }

    def start(self) -> None:
        """Start worker threads."""
        if self._running:
            return

        self._running = True
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"{self.name}-{i}"
            )
            worker.start()
            self._workers.append(worker)

    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Stop worker threads."""
        self._running = False

        for _ in self._workers:
            try:
                self._queue.put_nowait((float('inf'), 0, None))
            except queue.Full:
                pass

        if wait:
            deadline = time.time() + timeout
            for worker in self._workers:
                remaining = deadline - time.time()
                if remaining > 0:
                    worker.join(timeout=remaining)

        self._workers.clear()

    def submit(
        self,
        func: Callable,
        *args,
        priority: int = 0,
        **kwargs
    ) -> Future:
        """
        Submit work for async execution.

        Args:
            func: Function to call
            *args: Positional arguments
            priority: Lower = higher priority
            **kwargs: Keyword arguments

        Returns:
            Future for the result
        """
        work = WorkItem(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority
        )

        with self._lock:
            self._counter += 1
            counter = self._counter

        try:
            self._queue.put_nowait((priority, counter, work))
            self._stats['tasks_submitted'] += 1
        except queue.Full:
            work.future.set_exception(
                RuntimeError("Worker pool queue full")
            )

        return work.future

    def _worker_loop(self) -> None:
        """Worker loop."""
        while self._running:
            try:
                _, _, work = self._queue.get(timeout=1.0)

                if work is None:
                    break

                try:
                    result = work.func(*work.args, **work.kwargs)
                    work.future.set_result(result)
                    self._stats['tasks_completed'] += 1
                except Exception as e:
                    work.future.set_exception(e)
                    self._stats['tasks_failed'] += 1

            except queue.Empty:
                continue

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            'queue_size': self._queue.qsize(),
            'workers': len(self._workers),
            'running': self._running,
        }

    @property
    def is_running(self) -> bool:
        return self._running
