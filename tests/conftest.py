"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sine_samples():
    """Generate a 1-second 440Hz sine wave at 24kHz, int16."""
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    return sr, samples


@pytest.fixture
def silent_samples():
    """Generate 1 second of silence at 24kHz, int16."""
    sr = 24000
    return sr, np.zeros(sr, dtype=np.int16)


@pytest.fixture
def short_samples():
    """Generate a very short 0.05s tone."""
    sr = 24000
    n = int(sr * 0.05)
    t = np.linspace(0, 0.05, n, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    return sr, samples
