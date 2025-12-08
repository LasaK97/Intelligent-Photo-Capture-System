import structlog
from datetime import datetime, timezone
from pathlib import Path
import sys


def custom_console_renderer_call(self, logger, method_name, event_dict):
    """Custom __call__ method for ConsoleRenderer that puts filename first."""

    # Extract logger name and convert to filename
    logger_name = event_dict.pop('logger_name', None)
    if not logger_name:
        # Try to get from logger object
        logger_name = logger.name if hasattr(logger, 'name') else 'unknown'

    # Convert __main__ to actual script filename
    if logger_name == '__main__':
        if sys.argv and len(sys.argv) > 0:
            script_path = Path(sys.argv[0])
            filename = script_path.name if script_path.name else '__main__'
        else:
            filename = '__main__'
    elif '.' in str(logger_name):
        # Module path like 'scripts.yolo_to_tensorrt' -> 'yolo_to_tensorrt.py'
        filename = str(logger_name).split('.')[-1] + '.py'
    else:
        filename = str(logger_name) + '.py' if not str(logger_name).endswith('.py') else str(logger_name)

    # Extract other keys
    level = event_dict.pop('level', 'info')
    event = event_dict.pop('event', '')
    timestamp = event_dict.pop('timestamp', None)

    # Remove extra keys
    event_dict.pop('logger', None)
    event_dict.pop('method', None)

    # Format timestamp
    if timestamp:
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            timestamp_str = str(timestamp)
    else:
        timestamp_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # ANSI color codes
    colors = {
        'debug': '\033[36m',
        'info': '\033[32m',
        'warning': '\033[33m',
        'error': '\033[31m',
        'critical': '\033[31m\033[1m',
    }
    reset = '\033[0m'
    bold = '\033[1m'
    cyan = '\033[96m'  # Light blue/cyan color for filename

    level_color = colors.get(level, '')

    # Format: [filename] timestamp [level] event
    parts = [
        f"{cyan}[{filename}]{reset}",  # Filename in light blue
        timestamp_str,
        f"[{level_color}{bold}{level:9s}{reset}]",
        f"{bold}{event}{reset}"
    ]

    # Add remaining key-value pairs
    if event_dict:
        kv_parts = []
        for k, v in sorted(event_dict.items()):
            if not k.startswith('_'):
                kv_parts.append(f"{k}={v}")
        if kv_parts:
            parts.append(' '.join(kv_parts))

    return ' '.join(parts)


# Save the original __call__ method
_original_console_renderer_call = structlog.dev.ConsoleRenderer.__call__

# Replace the __call__ method on the ConsoleRenderer class
structlog.dev.ConsoleRenderer.__call__ = custom_console_renderer_call
