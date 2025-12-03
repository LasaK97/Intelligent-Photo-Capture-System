import asyncio
import threading
import time
import queue
import functools
from typing import Any, Callable, Optional, TypeVar, Generic, Union, List
from concurrent.futures import ThreadPoolExecutor, Future
import weakref
import psutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from .logger import get_logger
from .exceptions import ManriixError, TimeoutError, PerformanceError

logger = get_logger(__name__)

T = TypeVar("T")

class AsyncThreadSafeQueue(Generic[T]):
    """Thread safe-queue with async interface"""

    def __init__(self, maxsize: int = 0):
        self._queue: queue.Queue[T] = queue.Queue(maxsize=maxsize)
        self._getters: List[asyncio.Future] = []
        self._putters: List[asyncio.Future] = []
        self._maxsize = maxsize
        self._lock = threading.Lock()

    async def put(self, item: T, timeout: Optional[float] = None) -> None:
        """Put item into the queue"""
        loop = asyncio.get_running_loop()

        def _put():
            try:
                self._queue.put(item, block=True, timeout=timeout)
                return True
            except queue.Full:
                return False

        if timeout:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _put),
                timeout = timeout
            )
            if not result:
                raise queue.Full("Queue is full")
        else:
            await loop.run_in_executor(None, _put)

    async def get(self, timeout: Optional[float] = None) -> T:
        """Get item from the queue"""
        loop = asyncio.get_running_loop()

        def _get():
            return self._queue.get(block=True, timeout=timeout)

        try:
            if timeout:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, _get),
                    timeout = timeout
                )
            else:
                return await loop.run_in_executor(None, _get)
        except queue.Empty:
            raise TimeoutError("Queue get timeout")

    def put_nowait(self, item: T) -> None:
        """Put item into the queue"""
        self._queue.put_nowait(item)

    def get_nowait(self) -> T:
        """Get item from the queue"""
        return self._queue.get_nowait()

    @property
    def qsize(self) -> int:
        """get queue size"""
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """check if queue is empty"""
        return self._queue.empty()

    @property
    def full(self) -> bool:
        """check if queue is full"""
        return self._queue.full()

class PerformanceMonitor:
    """Monitor performance of operations"""

    def __init__(self, max_samples: int = 100) -> None:
        self.max_samples = max_samples
        self.samples: List[float] = []
        self.lock = threading.Lock()

    def add_sample(self, duration: float) -> None:
        """add performance sample"""
        with self.lock:
            self.samples.append(duration)
            if len(self.samples) >= self.max_samples:
                self.samples.pop(0)

    def get_stat(self) -> dict:
        """get performance statistics"""
        with self.lock:
            if not self.samples:
                return {'count': 0}

            avg = sum(self.samples) / len(self.samples)
            min_val = min(self.samples)
            max_val = max(self.samples)

            #percentiles
            sorted_samples = sorted(self.samples)
            p50 = sorted_samples[len(sorted_samples) // 2]
            p95 = sorted_samples[int(len(sorted_samples) * 0.95)]
            p99 = sorted_samples[int(len(sorted_samples) * 0.99)]

            return {
                'count': len(self.samples),
                'average_ms': round(avg * 1000, 2),
                'min_ms': round(min_val * 1000, 2),
                'max_ms': round(max_val * 1000, 2),
                'p50_ms': round(p50 * 1000, 2),
                'p95_ms': round(p95 * 1000, 2),
                'p99_ms': round(p99 * 1000, 2)
            }


class ThreadPoolManager:
    """Thread pool manager"""

    def __init__(self) -> None:
        settings = get_settings()
        self.max_workers = settings.threading.max_workers
        self.timeout = settings.threading.thread_timeout

        #seperate thread pools for different operations
        self.vision_pool = ThreadPoolExecutor(
            max_workers = max(2, self.max_workers // 2),
            thread_name_prefix="vision"
        )
        self.io_pool = ThreadPoolExecutor(
            max_workers = max(2, self.max_workers // 4),
            thread_name_prefix="io"
        )
        self.compute_pool = ThreadPoolExecutor(
            max_workers = max(2, self.max_workers // 4),
            thread_name_prefix="compute"
        )

        #performance monitor
        self.monitors = {
            'vision': PerformanceMonitor(),
            'io': PerformanceMonitor(),
            'compute': PerformanceMonitor()
        }

        logger.info("thread_pools_initialized", max_workers = self.max_workers)

    async def run_vision_task(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """run vision task"""
        return await self._run_with_monitoring('vision', self.vision_pool, func, *args, **kwargs)

    async def run_io_task(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """run io task"""
        return await self._run_with_monitoring('io', self.io_pool, func, *args, **kwargs)

    async def run_compute_task(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """run compute task"""
        return await self._run_with_monitoring('compute', self.compute_pool, func, *args, **kwargs)

    async def _run_with_monitoring(
            self,
            pool_name: str,
            pool: ThreadPoolExecutor,
            func: Callable,
            *args: Any,
            **kwargs: Any
        ) -> Any:
        """run with performance monitoring"""

        start_time = time.time()

        try:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(pool, functools.partial(func, *args, **kwargs))
            result = await asyncio.wait_for(future, self.timeout)

            duration = time.perf_counter() - start_time
            self.monitors[pool_name].add_sample(duration)

            logger.debug(
                "thread_task_completed",
                pool = pool_name,
                duration_ms = round(duration * 1000, 2),
                function = func.__name__,
            )
            return result
        except asyncio.TimeoutError:
            duration = time.perf_counter() - start_time
            logger.error(
                "thread_task_timeout",
                pool = pool_name,
                duration_ms = round(duration * 1000, 2),
                timeout_ms = self.timeout * 1000,
                function = func.__name__,
            )
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(
                "thread_task_error",
                pool=pool_name,
                duration_ms=round(duration * 1000, 2),
                function=func.__name__,
                error=str(e)
            )
            raise

    def get_performance_stats(self) -> dict:
        """get performance statistics"""
        return {
            pool_name: monitor.get_stat() for pool_name, monitor in self.monitors.items()
        }

    def shutdown(self, wait: bool = True) -> None:
        """shutdown all threads"""
        logger.info("shutdown_thread_pools")

        self.vision_pool.shutdown(wait=wait)
        self.io_pool.shutdown(wait=wait)
        self.compute_pool.shutdown(wait=wait)

        logger.info("thread_pools_shutdown_completed")

#global thread manager
_thread_manager: Optional[ThreadPoolManager] = None

def get_thread_manager() -> ThreadPoolManager:
    """get thread pool manager"""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = ThreadPoolManager()
    return _thread_manager

def async_retry(
        max_attempts: int = 3,
        delay: float = 1.0,
        exponential_backoff: bool = True,
        exceptions: tuple = (Exception,),
):
    """retry async function with exponential backoff"""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts = max_attempts,
                            final_error=str(e),
                        )

                    wait_time = delay * (2 ** attempt) if exponential_backoff else delay

                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt = attempt +1,
                        max_attempts = max_attempts,
                        error = str(e),
                        wait_time = wait_time,
                    )

                    await asyncio.sleep(wait_time)

            raise last_exception
        return wrapper
    return decorator

def sync_to_async(func: Callable) -> Callable:
    """convert sync function to async function"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    return wrapper

class AsyncLock:
    """Async lock implementation"""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


class ResourceMonitor:
    """monitor resource usage"""

    def __init__(self):
        self.process = psutil.Process()
        self.settings = get_settings()

    def check_resources(self) -> dict:
        """check resource usage"""

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024 ** 3),
            'memory_available_gb': memory.available / (1024 ** 3),
            'process_memory_mb': process_memory.rss / (1024 ** 2)
        }


    def check_performance_limits(self) -> bool:
        """check performance limits"""

        stats = self.check_resources()

        if stats['cpu_percent'] > self.settings.performance.max_cpu_usage:
            logger.warning(
                "cpu_high_usage",
                cpu_percent = stats['cpu_percent'],
            )
            return False

        if stats['memory_percent'] > 90: #critical threshold
            logger.warning(
                "memory_high_critical",
                memory_percent = stats['memory_percent'],
            )
            return False

        if stats['process_memory_mb'] > self.settings.performance.max_memory_usage * 1024:
            logger.warning(
                "high_process_memory_usage",
                memory_percent = stats['process_memory_mb'],
            )
            return False

        return True






