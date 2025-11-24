"""
Groq title generation client utilities.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict

try:  # pragma: no cover - import guarded for tests
    from groq import Groq
except ImportError:  # pragma: no cover - tests can monkeypatch get_groq_client
    Groq = None  # type: ignore[assignment]


def get_groq_client() -> "Groq":
    """
    Return a Groq client instance using GROQ_API_KEY env var.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GROQ_API_KEY environment variable not set")
    if Groq is None:  # pragma: no cover - exercised in live environment
        raise ImportError(
            "groq package is not installed. Install with `pip install groq`."
        )
    return Groq(api_key=api_key)


def generate_title_from_transcript(
    transcript_json: Dict[str, Any], system_prompt: str
) -> str:
    """
    Generate YouTube title from transcript using Groq.

    Args:
        transcript_json: Loaded transcript JSON (with 'text' field)
        system_prompt: System instructions for title generation

    Returns:
        Generated title (string, 30-80 chars, validated)

    Raises:
        ValueError: If transcript has no text
        RuntimeError: If LLM returns invalid title after retry
    """
    transcript_text = _transcript_to_text(transcript_json)
    user_prompt = _build_user_prompt(transcript_text)

    title = _request_title(system_prompt, user_prompt)
    is_valid, error = _validate_title(title)
    if is_valid:
        return title

    retry_prompt = (
        f"{user_prompt}\n\n"
        f"Previous attempt was invalid: {error}. "
        "Respond with a single improved title that follows every rule. "
        "Do not add quotes or commentary."
    )
    retry_title = _request_title(system_prompt, retry_prompt)
    is_valid_retry, retry_error = _validate_title(retry_title)
    if is_valid_retry:
        return retry_title

    raise RuntimeError(
        f"Groq returned invalid title after retry: {retry_error}. "
        f"Last candidate: {retry_title!r}"
    )


def _request_title(system_prompt: str, user_prompt: str) -> str:
    """
    Call Groq API and return cleaned title text.
    """
    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=100,
    )

    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("Groq response contained no choices.")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        raise RuntimeError("Groq response missing textual content.")

    return _clean_title_text(content)


def _build_user_prompt(transcript_text: str) -> str:
    """
    Build the user prompt for the LLM.
    """
    return (
        "Transcript of a trading clip:\n\n"
        f"{transcript_text}\n\n"
        "Generate a YouTube Shorts title (30-80 characters).\n\n"
        "Output ONLY the title text, nothing else. No quotes, no explanation, just the title."
    )


def _clean_title_text(raw: str) -> str:
    """
    Strip whitespace and surrounding quotes from LLM output.
    """
    text = raw.strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1]
    return text.strip()


def _validate_title(title: str) -> tuple[bool, str]:
    """
    Validate a generated title against length and emoji rules.
    """
    length = len(title)
    if length < 30:
        return False, f"Too short ({length} chars, need 30+)"
    if length > 80:
        return False, f"Too long ({length} chars, max 80)"

    emoji_pattern = re.compile(
        "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "]+", flags=re.UNICODE
    )
    if emoji_pattern.search(title):
        return False, "Contains emojis"

    return True, "Valid"


def _transcript_to_text(transcript_json: Dict[str, Any]) -> str:
    """
    Extract plain text from transcript JSON.
    """
    text_field = transcript_json.get("text")
    if isinstance(text_field, str) and text_field.strip():
        return text_field.strip()

    words = transcript_json.get("words")
    if isinstance(words, list):
        tokens = [w.get("word", "").strip() for w in words if isinstance(w, dict)]
        combined = " ".join(token for token in tokens if token).strip()
        if combined:
            return combined

    segments = transcript_json.get("segments")
    if isinstance(segments, list) and segments:
        segment_texts = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seg_text = seg.get("text")
            if isinstance(seg_text, str):
                segment_texts.append(seg_text.strip())
        combined = " ".join(text for text in segment_texts if text).strip()
        if combined:
            return combined

    raise ValueError("No text found in transcript JSON")


__all__ = [
    "generate_title_from_transcript",
    "_transcript_to_text",
    "_validate_title",
]
