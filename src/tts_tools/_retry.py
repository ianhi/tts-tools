"""Simple retry logic for sync and async callables."""

from __future__ import annotations

import asyncio
import time
from typing import TypeVar

T = TypeVar("T")


def retry_sync(fn, *, max_retries: int = 3, backoff: float = 1.0) -> T:
    """Call *fn()* up to *max_retries* times, with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff * (2**attempt))
    raise last_exc  # type: ignore[misc]


async def retry_async(fn, *, max_retries: int = 3, backoff: float = 1.0) -> T:
    """Await *fn()* up to *max_retries* times, with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff * (2**attempt))
    raise last_exc  # type: ignore[misc]
