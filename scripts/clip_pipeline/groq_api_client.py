#!/usr/bin/env python3
"""
groq_api_client.py - Groq Llama 3.3 API client (Production Version)

Drop-in replacement for the Gemini client used in the clipping pipeline.
Handles both context generation and clip selection with hook candidates.

Requirements:
    pip install groq

Environment:
    export GROQ_API_KEY="your-key-here"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from groq import Groq
except ImportError:
    print("ERROR: Missing groq package. Install with: pip install groq", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"


# ============================================================================
# Groq Client
# ============================================================================

class GroqClient:
    """Groq chat-completions client."""

    def __init__(self, api_key: Optional[str] = None, model: str = MODEL_NAME):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "Get one from: https://console.groq.com/\n"
                "Then: export GROQ_API_KEY='your-key-here'"
            )

        self.model_name = model
        self.client = Groq(api_key=self.api_key)
        print(f"✓ Groq client initialized: {self.model_name}")

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
    ) -> str:
        """Generate content using Groq chat completions."""

        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
        except Exception as exc:
            print(f"ERROR: Groq API call failed: {exc}", file=sys.stderr)
            raise

        choices = getattr(response, "choices", None)
        if not choices:
            return ""

        message = choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)
        if not isinstance(content, str):
            return ""

        return content.strip()

    def count_tokens(self, text: str) -> int:
        """Approximate token count (Groq SDK lacks tokenizer)."""
        if not text:
            return 0
        return max(1, len(text) // 4)


# ============================================================================
# Helper Functions
# ============================================================================

def load_transcript_text(transcript_path: str) -> tuple[str, List[Dict[str, Any]], float]:
    """
    Load transcript and format as plain text.

    Returns:
        (formatted_text, segments_list, duration_seconds)
    """
    with open(transcript_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
    elif isinstance(data, list):
        segments = data
    else:
        raise ValueError(f"Unexpected transcript format in {transcript_path}")

    if not segments:
        raise ValueError(f"No segments found in {transcript_path}")

    text_lines: List[str] = []
    for seg in segments:
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()
        if text:
            text_lines.append(f"[{start:.1f}s] {text}")

    formatted_text = "\n".join(text_lines)
    duration = segments[-1].get("end", 0) if segments else 0

    return formatted_text, segments, duration


def clean_yaml_response(response: str) -> str:
    """Clean LLM response to extract pure YAML."""
    response = response.strip()

    if response.startswith("```yaml"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    return response.strip()


# ============================================================================
# Pipeline Functions
# ============================================================================

def generate_context(transcript_path: str, output_path: str) -> bool:
    """Generate context.yaml from transcript using Groq."""

    print("\n" + "─" * 60)
    print("GENERATING CONTEXT (Groq)")
    print("─" * 60)

    try:
        transcript_text, segments, duration = load_transcript_text(transcript_path)
    except Exception as exc:
        print(f"❌ Failed to load transcript: {exc}")
        return False

    duration_min = duration / 60
    print(f"  Transcript: {len(segments)} segments, {duration_min:.1f} min")

    system_prompt = "# SYSTEM_PROMPT_PLACEHOLDER_CONTEXT_GEN"

    client = GroqClient()
    token_count = client.count_tokens(transcript_text)
    print(f"  Tokens (est): {token_count:,}")
    print("  Calling Groq llama-3.3-70b-versatile...")

    try:
        response = client.generate(
            prompt=transcript_text,
            system_instruction=system_prompt,
            temperature=0.7,
            max_output_tokens=4096,
        )
    except Exception as exc:
        print(f"❌ API call failed: {exc}")
        return False

    cleaned_response = clean_yaml_response(response)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(cleaned_response)

    print(f"  ✓ Saved to {output_path}")
    return True


def select_clips(
    transcript_path: str,
    context_path: str,
    hook_candidates_path: str,
    output_path: str,
) -> bool:
    """Select clips using Groq with hook candidate context."""

    print("\n" + "─" * 60)
    print("SELECTING CLIPS (Groq)")
    print("─" * 60)

    try:
        transcript_text, segments, _ = load_transcript_text(transcript_path)
    except Exception as exc:
        print(f"❌ Failed to load transcript: {exc}")
        return False

    if not Path(hook_candidates_path).exists():
        print(f"❌ Hook candidates not found: {hook_candidates_path}")
        return False

    with open(hook_candidates_path, "r", encoding="utf-8") as handle:
        candidate_data = json.load(handle)

    candidates = candidate_data.get("candidates", [])[:20]
    print(f"  Loaded {len(candidates)} hook candidates")

    with open(context_path, "r", encoding="utf-8") as handle:
        context_text = handle.read()

    system_prompt = "# SYSTEM_PROMPT_PLACEHOLDER_CLIP_SELECT"

    user_prompt = [
        "EPISODE CONTEXT:",
        context_text.strip(),
        "",
        "HOOK CANDIDATES (Top 20):",
    ]

    for idx, candidate in enumerate(candidates, 1):
        start_time = candidate.get("start", 0)
        end_time = start_time + 90
        context_window = []
        for seg in segments:
            seg_start = seg.get("start", 0)
            if start_time <= seg_start < end_time:
                context_window.append(f"[{seg_start:.1f}s] {seg.get('text', '').strip()}")
        signals = candidate.get("signals", [])
        user_prompt.append(
            f"\nCANDIDATE #{idx} (Score: {candidate.get('hook_score', 'n/a')}/10)\n"
            f"Opening: \"{candidate.get('text', '').strip()}\"\n"
            f"Signals: {', '.join(signals) if signals else 'n/a'}\n"
            f"Context Window (next 90s):\n" + "\n".join(context_window[:30])
        )

    user_prompt.extend(
        [
            "",
            "FULL TRANSCRIPT:",
            transcript_text,
            "",
            "Select 5-10 clips that pair the strongest hooks with satisfying payoffs.",
        ]
    )

    prompt_text = "\n".join(user_prompt)

    client = GroqClient()
    token_count = client.count_tokens(prompt_text)
    print(f"  Input tokens (est): {token_count:,}")
    print("  Calling Groq llama-3.3-70b-versatile...")

    try:
        response = client.generate(
            prompt=prompt_text,
            system_instruction=system_prompt,
            temperature=0.5,
            max_output_tokens=8192,
        )
    except Exception as exc:
        print(f"❌ LLM call failed: {exc}")
        return False

    cleaned_response = clean_yaml_response(response)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(cleaned_response)

    print(f"  ✓ Selected clips saved to {output_path}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("GROQ CLIENT TEST")
    print("=" * 60)
    try:
        client = GroqClient()
        response = client.generate(
            prompt="Say hello and confirm you are Groq llama-3.3-70b.",
            system_instruction="You are a friendly assistant.",
            temperature=0.7,
            max_output_tokens=64,
        )
        print(f"\nResponse:\n{response}")
    except Exception as exc:
        print(f"❌ Test failed: {exc}")
        sys.exit(1)

    print("\n✅ Client test complete!")
    print("=" * 60)
