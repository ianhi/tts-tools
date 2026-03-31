"""Tests for retry logic."""

import asyncio

import pytest

from tts_tools._retry import retry_async, retry_sync


def test_retry_sync_succeeds_first_try():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result = retry_sync(fn, max_retries=3, backoff=0.01)
    assert result == "ok"
    assert len(calls) == 1


def test_retry_sync_succeeds_after_failures():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "ok"

    result = retry_sync(fn, max_retries=3, backoff=0.01)
    assert result == "ok"
    assert len(calls) == 3


def test_retry_sync_exhausted():
    def fn():
        raise ValueError("always fails")

    with pytest.raises(ValueError, match="always fails"):
        retry_sync(fn, max_retries=2, backoff=0.01)


@pytest.mark.asyncio
async def test_retry_async_succeeds():
    calls = []

    async def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("not yet")
        return "ok"

    result = await retry_async(fn, max_retries=3, backoff=0.01)
    assert result == "ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_retry_async_exhausted():
    async def fn():
        raise ValueError("always fails")

    with pytest.raises(ValueError, match="always fails"):
        await retry_async(fn, max_retries=2, backoff=0.01)
