"""Language-aware sentence punctuation."""

from __future__ import annotations

# Language codes that use a non-ASCII full stop
_STOPS: dict[str, str] = {
    "bn-IN": "\u0964",  # Bengali danda ।
}

_DEFAULT_STOP = "."


def ensure_sentence_stop(text: str, language: str) -> str:
    """Append a sentence-ending stop if the text doesn't already have one."""
    stop = _STOPS.get(language, _DEFAULT_STOP)
    if text.rstrip().endswith(stop) or text.rstrip().endswith("."):
        return text
    return text + stop
