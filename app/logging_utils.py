import logging
import time
from collections import deque

recent_logs = deque(maxlen=200)


def log_event(message, level="info"):
    ts = int(time.time())
    entry = {"ts": ts, "message": message, "level": level}
    recent_logs.appendleft(entry)
    getattr(logging, level, logging.info)(message)

