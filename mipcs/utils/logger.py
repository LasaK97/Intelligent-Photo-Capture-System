from . import structlog_patch

import asyncio
import logging
import logging.handlers
import json
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import structlog

from structlog.types import FilteringBoundLogger

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from .exceptions import ManriixError

class PerformanceTimer:
    """Context manager for performance timing"""

    def __init__(self, logger: FilteringBoundLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.debug("operation_started", operation=self.operation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.logger.info(
            "operation_completed",
            operation=self.operation,
            duration_ms=round(duration * 1000, 2),
            success=exc_type is None
        )

        if exc_type:
            self.logger.error(
                "operation_failed",
                operation=self.operation,
                error_type=exc_type.__name__,
                error_message=str(exc_val) if exc_val else None,
                duration_ms=round(duration * 1000, 2)
            )

class AsyncLogHandler(logging.Handler):
    """Async logging handler for non blocking logging"""

    def __init__(self, handler: logging.Handler):
        super().__init__()
        self.handler = handler
        self.queue = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        """Start the async logging task"""
        self.task = asyncio.create_task(self._log_worker())

    async def stop(self):
        """Stop the async logging task"""
        self._stop_event.set()
        if self.task:
            await self.task

    def emit(self, record):
        """Emit log record asynchronously"""
        try:
            self.queue.put_nowait(record)
        except asyncio.QueueFull:
            pass

    async def _log_worker(self):
        """Log worker task"""
        while not self._stop_event.is_set():
            try:
                record = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.handler.emit(record)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Logging error: {e}", file=sys.stderr)


def add_context_processor(logger, method_name, event_dict):
    """Add contextual information to log records."""
    event_dict['timestamp'] = datetime.utcnow().isoformat()
    event_dict['logger_name'] = logger.name if hasattr(logger, 'name') else 'unknown'
    event_dict['method'] = method_name
    return event_dict


def add_performance_processor(logger, method_name, event_dict):
    """Add performance metrics to log records."""
    import psutil

    # Add system performance metrics for important events
    if event_dict.get('event') in ['operation_completed', 'model_inference', 'frame_processed']:
        event_dict['system_metrics'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_mb': psutil.virtual_memory().available // (1024 * 1024)
        }

    return event_dict


def exception_processor(logger, method_name, event_dict):
    """Process exceptions for structured logging."""
    if 'exception' in event_dict:
        exc = event_dict['exception']
        if isinstance(exc, ManriixError):
            event_dict.update(exc.to_dict())
        elif isinstance(exc, Exception):
            event_dict['exception_type'] = exc.__class__.__name__
            event_dict['exception_message'] = str(exc)

    return event_dict

def setup_logging() -> FilteringBoundLogger:
    """Setup logging configuration"""
    settings = get_settings()

    #log dir
    log_path = Path(settings.logging.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    #confi stdlib logging
    stdlib_logger = logging.getLogger()
    stdlib_logger.setLevel(getattr(logging, settings.logging.level))

    #remove existing handlers
    for handler in stdlib_logger.handlers[:]:
        stdlib_logger.removeHandler(handler)

    #file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.logging.file_path,
        maxBytes=settings.logging.max_file_size_mb * 1024 * 1024,
        backupCount=settings.logging.backup_count,
        encoding='utf-8'
    )

    file_handler.setLevel(getattr(logging, settings.logging.level))

    #console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.logging.level))

    #configure formatters
    if settings.logging.format == 'structured':
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    #add handlers
    stdlib_logger.addHandler(file_handler)
    if settings.logging.console_output:
        stdlib_logger.addHandler(console_handler)

    logging.root.setLevel(getattr(logging, settings.logging.level))
    for name in list(logging.Logger.manager.loggerDict):
        logging.getLogger(name).setLevel(getattr(logging, settings.logging.level))

    #config structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            add_context_processor,
            add_performance_processor,
            exception_processor,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.logging.format == "structured"
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False
    )

    return structlog.get_logger(logger_name=settings.system.name)

def get_logger(name: str = None) -> FilteringBoundLogger:
    """Get logger instance"""
    _settings = get_settings()
    logger_name = name or _settings.system.name
    stdlib_logger = logging.getLogger(logger_name)
    stdlib_logger.setLevel(getattr(logging, _settings.logging.level))

    return structlog.get_logger(logger_name=logger_name)

def log_performance(operation: str):
    """Decorator for performance logging"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            with PerformanceTimer(logger, operation):
                return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            with PerformanceTimer(logger, operation):
                return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

#global logger instance
_logger: Optional[FilteringBoundLogger] = None

def init_logging() -> FilteringBoundLogger:
    """Initialize logging system"""
    global _logger
    _logger = setup_logging()
    _logger.info("logging_system_initialized")
    return _logger

def get_global_logger() -> FilteringBoundLogger:
    """Get global logger"""
    global _logger
    if _logger is None:
        _logger = init_logging()
    return _logger