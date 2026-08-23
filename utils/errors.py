"""
Human-readable error presentation for Synapse.

Maps the ExtractionError.stage taxonomy (defined in models/schemas.py,
produced by services/gemini_service.py) to plain-language messages and
optional troubleshooting hints. Centralized here so every place that
shows an extraction error (transcript form, audio form, future
recommendation panel) presents it consistently instead of each
re-implementing its own if/elif on `stage`.

Never surfaces raw exception text, stack traces, or API internals to
the user — that stays in ExtractionError.raw_response / server logs
only, for developer diagnosis.
"""

from __future__ import annotations

from typing import Optional

from models.schemas import ExtractionError

_HINTS: dict[str, str] = {
    "DATA_ERROR": "Check that you provided real, substantive content before submitting.",
    "AI_API_ERROR": "Check that GEMINI_API_KEY is configured correctly (Streamlit secrets, or your local .env).",
    "AI_OUTPUT_ERROR": "The AI didn't return a structured result, even after a retry. Try again, or simplify the input.",
    "VALIDATION_ERROR": "The AI's response didn't match the expected format, even after a retry.",
}

_DEFAULT_HINT = "If this keeps happening, try again in a moment."


def friendly_hint(error: ExtractionError) -> Optional[str]:
    """A short troubleshooting hint appropriate to the error's stage.
    Returns None if there's nothing more useful to say than the
    error's own message."""
    return _HINTS.get(error.stage, _DEFAULT_HINT)
