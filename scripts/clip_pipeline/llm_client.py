#!/usr/bin/env python3
"""
llm_client.py - Gemini 2.5 Flash API client (Production Version)

Optimized for sentence-level transcripts with hook-first architecture.
Ultra-cheap: ~$0.01 per 2-hour video with 1M token context window.

Requirements:
    pip install google-generativeai

Environment:
    export GOOGLE_API_KEY="your-key-here"
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: Missing google-generativeai. Install with: pip install google-generativeai", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"  # 1M context, $0.15/$0.60 per 1M tokens


# ============================================================================
# Gemini Client
# ============================================================================

class GeminiClient:
    """Google Gemini 2.5 Flash API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = MODEL_NAME):
        self.api_key = api_key or GOOGLE_API_KEY
        self.model_name = model
        
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set.\n"
                "Get one from: https://aistudio.google.com/apikey\n"
                "Then: export GOOGLE_API_KEY='your-key-here'"
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        print(f"✓ Gemini client initialized: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
    ) -> str:
        """Generate content using Gemini 2.5 Flash."""
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            if system_instruction:
                model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction,
                    safety_settings=safety_settings
                )
            else:
                model = genai.GenerativeModel(
                    self.model_name,
                    safety_settings=safety_settings
                )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            
            if not response.parts:
                return "BLOCKED_BY_SAFETY_FILTER"
            
            return response.text
        
        except Exception as e:
            print(f"ERROR: Gemini API call failed: {e}", file=sys.stderr)
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's tokenizer."""
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception as e:
            print(f"WARNING: Token counting failed, using estimation: {e}", file=sys.stderr)
            return len(text) // 4


# ============================================================================
# Helper Functions
# ============================================================================

def load_transcript_text(transcript_path: str) -> tuple[str, List[Dict], float]:
    """
    Load transcript and format as plain text.
    
    Returns:
        (formatted_text, segments_list, duration_seconds)
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
    elif isinstance(data, list):
        segments = data
    else:
        raise ValueError(f"Unexpected transcript format in {transcript_path}")
    
    if not segments:
        raise ValueError(f"No segments found in {transcript_path}")
    
    text_lines = []
    for seg in segments:
        start = seg.get('start', 0)
        text = seg.get('text', '').strip()
        if text:
            text_lines.append(f"[{start:.1f}s] {text}")
    
    formatted_text = "\n".join(text_lines)
    duration = segments[-1].get('end', 0) if segments else 0
    
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
    """Generate context.yaml from transcript using Gemini 2.5 Flash."""
    
    print("\n" + "─"*60)
    print("GENERATING CONTEXT")
    print("─"*60)
    
    try:
        transcript_text, segments, duration = load_transcript_text(transcript_path)
    except Exception as e:
        print(f"❌ Failed to load transcript: {e}")
        return False
    
    duration_min = duration / 60
    print(f"  Transcript: {len(segments)} segments, {duration_min:.1f} min")
    
    system_prompt = """You are an expert podcast analyst. Analyze this transcript and extract:

1. **Key themes** - Main topics discussed with why they matter
2. **Memorable moments** - Specific timestamps of powerful/interesting moments
3. **Guest info** - Names, credentials, expertise
4. **Episode metadata** - Title, tone, domain, summary

Output as **valid YAML** in this exact format:
```yaml
podcast: "Podcast Name"
episode_title: "Episode Title"
guests: ["Guest Name"]
domain: "domain/topic"
tone: "warm, reflective, evidence-based"
one_sentence_summary: "Concise episode summary"
key_themes:
  - name: "Theme name"
    why_it_matters: "Why this matters to audience"
    representative_moments:
      - time: 123.4
        note: "What happens at this moment"
memorable_lines:
  - time: 789.0
    line: "Exact memorable quote"
```

Be concise, insightful, and focus on what makes this content unique."""
    
    client = GeminiClient()
    token_count = client.count_tokens(transcript_text)
    cost_estimate = (token_count / 1_000_000) * 0.15
    
    print(f"  Tokens: {token_count:,} (~${cost_estimate:.4f})")
    
    if token_count > 1_000_000:
        print(f"  ⚠️  WARNING: Exceeds 1M token limit!")
        return False
    
    print(f"  Calling Gemini 2.5 Flash...")
    
    try:
        response = client.generate(
            prompt=transcript_text,
            system_instruction=system_prompt,
            temperature=0.7,
            max_output_tokens=4096
        )
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False
    
    cleaned_response = clean_yaml_response(response)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_response)
    
    print(f"  ✓ Saved to {output_path}")
    return True


def select_clips_from_candidates(
    candidates_path: str,
    transcript_path: str,
    context_path: str,
    output_path: str
) -> bool:
    """
    Select clips from pre-scored hook candidates.
    
    Args:
        candidates_path: Path to hook_candidates.json
        transcript_path: Path to transcript JSON
        context_path: Path to context YAML
        output_path: Path to save clip selection YAML
    
    Returns:
        True if successful
    """
    client = GeminiClient()
    
    print("\n" + "─"*60)
    print("SELECTING CLIPS FROM CANDIDATES")
    print("─"*60)
    
    # Load candidates
    with open(candidates_path, 'r') as f:
        cand_data = json.load(f)
    
    candidates = cand_data["candidates"][:20]  # Top 20 only
    
    print(f"  Loaded {len(candidates)} pre-scored candidates")
    print(f"  Score range: {cand_data['metadata']['min_score']}-{cand_data['metadata']['max_score']}/10")
    
    # Load full transcript
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    segments = transcript_data.get("segments", transcript_data)
    
    # Load context
    with open(context_path, 'r') as f:
        context_text = f.read()
    
    # System prompt
    system_prompt = """You are refining PRE-SCORED hook candidates (scored 6-9/10 by algorithm).

Your job:
1. Verify hook is actually strong in context
2. Find natural ending 30-90 seconds after hook
3. Create compelling title

**CRITICAL:** These hooks were pre-validated for:
- Strong opening sentences
- No mid-conversation starts
- Self-contained beginnings

**YOU MUST:**
- Verify the hook delivers on its promise
- Find a satisfying conclusion within 30-90s
- Ensure clip makes sense standalone

**OUTPUT YAML:**
```yaml
- opening_sentence: "Exact text from candidate"
  closing_sentence: "Natural ending from context"
  hook_type: "Contrarian/Question/Warning/etc"
  hook_strength: 8
  title: "Title"
  tag: "Story/Framework/Contrarian"
  why_chosen: "Why this hook+payoff works"
```

**REJECT IF:**
- Hook misleading with full context
- No clear payoff in 90s
- Needs prior knowledge to understand

Select 5-10 best clips. Copy opening_sentence EXACTLY."""
    
    # Build user prompt with candidates
    user_prompt = f"""CONTEXT:
{context_text}

PRE-SCORED HOOK CANDIDATES (Top 20):

"""
    
    for i, cand in enumerate(candidates, 1):
        # Get 90s context window
        start_time = cand["start"]
        end_time = start_time + 90
        
        context_window = []
        for seg in segments:
            if start_time <= seg["start"] < end_time:
                context_window.append(f"[{seg['start']:.1f}s] {seg['text']}")
        
        user_prompt += f"""
CANDIDATE #{i} (Score: {cand['hook_score']}/10)
Opening: "{cand['text']}"
Signals: {', '.join(cand['signals'])}

Context (90s after hook):
{chr(10).join(context_window[:30])}

---
"""
    
    user_prompt += "\nSelect 5-10 best clips with strongest hook+payoff combination."
    
    # Count tokens
    input_tokens = client.count_tokens(user_prompt)
    print(f"  Input tokens: {input_tokens:,}")
    print(f"  Estimated cost: ~${(input_tokens / 1_000_000) * 0.15:.4f}")
    
    print(f"  Calling Gemini 2.5 Flash...")
    
    # Generate
    try:
        response = client.generate(
            prompt=user_prompt,
            system_instruction=system_prompt,
            temperature=0.7,
            max_output_tokens=8192
        )
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return False
    
    # Clean and save
    cleaned = clean_yaml_response(response)
    
    with open(output_path, 'w') as f:
        f.write(cleaned)
    
    print(f"  ✓ Selected clips saved to {output_path}")
    return True


def select_clips(
    transcript_path: str,
    context_path: str,
    hook_candidates_path: str,
    output_path: str
) -> bool:
    """
    Compatibility wrapper for orchestrate.py expecting select_clips signature.
    """
    return select_clips_from_candidates(
        hook_candidates_path,
        transcript_path,
        context_path,
        output_path,
    )


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("GEMINI 2.5 FLASH CLIENT TEST")
    print("="*60)
    
    try:
        client = GeminiClient()
        
        print("\nTesting basic generation...")
        response = client.generate(
            prompt="Say hello and confirm you are Gemini 2.5 Flash!",
            temperature=0.7,
            max_output_tokens=100
        )
        print(f"\nResponse:\n{response}")
        
        print("\n" + "─"*60)
        print("Testing token counting...")
        test_text = "This is a test sentence. " * 100
        tokens = client.count_tokens(test_text)
        chars = len(test_text)
        ratio = chars / tokens if tokens > 0 else 0
        print(f"  Text: {chars} chars")
        print(f"  Tokens: {tokens}")
        print(f"  Ratio: {ratio:.2f} chars/token")
        
        print("\n✅ Client test complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
