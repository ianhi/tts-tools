"""Tests for language-aware punctuation."""

from tts_tools._punctuation import ensure_sentence_stop


def test_bengali_adds_danda():
    assert ensure_sentence_stop("কথা", "bn-IN") == "কথা।"


def test_bengali_no_double_danda():
    assert ensure_sentence_stop("কথা।", "bn-IN") == "কথা।"


def test_english_adds_period():
    assert ensure_sentence_stop("hello", "en-US") == "hello."


def test_english_no_double_period():
    assert ensure_sentence_stop("hello.", "en-US") == "hello."


def test_spanish_adds_period():
    assert ensure_sentence_stop("hola", "es-US") == "hola."


def test_unknown_language_defaults_to_period():
    assert ensure_sentence_stop("test", "xx-XX") == "test."
