import structlog
from datetime import datetime, timezone


def custom_console_renderer_call(self, logger, method_name, event_dict):
    """Custom __call__ method for ConsoleRenderer with clean formatting."""

    # Extract keys
    level = event_dict.pop('level', 'info')
    event = event_dict.pop('event', '')
    timestamp = event_dict.pop('timestamp', None)

    # Remove extra keys
    event_dict.pop('logger_name', None)
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

    level_color = colors.get(level, '')

    # Format: timestamp [level] event
    parts = [
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