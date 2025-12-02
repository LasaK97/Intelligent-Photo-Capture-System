import asyncio
import threading
import time
import queue
import functools
from typing import Any, Callable, Optional, TypeVar, Generic, Union, List
from concurrent.futures import ThreadPoolExecutor, Future
import weakref

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
