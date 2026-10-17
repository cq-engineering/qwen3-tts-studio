"""Shared exceptions.

`_Cancelled` is raised inside a running job when a cancel is requested and
caught by the job worker, which then finalises the cancelled state. It lives
in its own module so both the writers layer and the jobs layer can import it
without a circular dependency.
"""
from __future__ import annotations


class _Cancelled(Exception):
    """Raised (and swallowed) when a job is cancelled mid-flight."""
