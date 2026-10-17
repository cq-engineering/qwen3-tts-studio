"""Rotating error log (errors.log).

Persistent, timestamped, rotation-capped error reporting. Server stdout
(server.log) is truncated on every start, so this is the durable record of
every 500, job failure and browser-side error.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from app.config import ERROR_LOG

_log = logging.getLogger("qwen3-tts_studio")
_log.setLevel(logging.INFO)
if not _log.handlers:
    _fh = RotatingFileHandler(ERROR_LOG, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _log.addHandler(_fh)
