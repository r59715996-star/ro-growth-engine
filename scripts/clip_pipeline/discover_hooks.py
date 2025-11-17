#!/usr/bin/env python3
"""
discover_hooks.py - Research-optimized hook scoring system v3

Scores transcript segments on hook quality (1-10) using viral hook research.
Filters weak intros, outputs top candidates for LLM refinement.

VERSION 3 IMPROVEMENTS:
- Incomplete sentence detection and penalty
- Refined contrarian detection (assertions vs explanatory)
- Context-dependent short statement penalty
- Demoted generic/weak questions to neutral

Usage:
    python scripts/discover_hooks.py

Output:
    plans/hook_candidates.json (top scoring segments)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


# ============================================================================
# CONFIGURATION
# ============================================================================

TRANSCRIPT_PATH = Path(os.environ.get("TRANSCRIPT_PATH", "data/transcripts/transcript2.json"))
OUTPUT_PATH = Path(os.environ.get("HOOK_CANDIDATES_PATH", "plans/hook_candidates.json"))

MIN_HOOK_SCORE = 6      # Minimum score to be considered
MAX_CANDIDATES = 30     # Maximum candidates to pass to LLM
TIER1_SOFT_CAP = 5      # Max Tier 1 contribution unless 3+ distinct


# ============================================================================
# HOOK PATTERNS (Research-Backed)
# ============================================================================

# Power words by category
POWER_WORDS = {
    "curiosity": [
        "secret", "truth", "nobody knows", "what you didn't know",
        "reveals", "hidden", "insider", "turns out", "actually"
    ],
    "negative": [
        "stop", "never", "worst", "hate", "biggest mistake",
        "fail", "avoid", "killing", "destroying", "ruining"
    ],
    "urgency": [
        "right now", "immediately", "before it's too late",
        "this second", "today", "urgent", "breaking"
    ],
    "emotion": [
        "shocking", "incredible", "unbelievable", "insane",
        "crazy", "terrifying", "devastating", "mind-blowing"
    ]
}

# Tier 1 Patterns (Scroll-Stoppers)
TIER1_PATTERNS = {
    "open_loop": [
        r"not what you (think|expect|thought|imagined)",
        r"you('ll| will) never (guess|believe|expect|imagine)",
        r"the (real|hidden|true|secret) reason",
        r"wait until you (hear|see|find out|learn)",
        r"but here's the (twist|catch|thing|kicker)",
    ],
    "contrarian_assertion": [
        r"^(most|all|everyone|everything) .* (is|are) (wrong|false|a lie|a myth)",
        r"^that's (wrong|false|a lie|not true|a myth)",
        r"^(you're|we're|they're) (doing|getting|thinking) .* wrong",
        r".* (is|are) (a lie|false|misleading|a scam|a hoax)",
        r"^(most|all) .* (don't|won't|can't|never)",
    ],
    "strong_address": [
        r"^(if|when) you('ve| have) (ever|always|never)",
        r"^you (need to|must|have to|should|can't)",
        r"^(are|do|have) you (making|doing|saying|struggling)",
        r"^your .* (is|are) (wrong|failing|killing|destroying)",
    ],
    "strong_question": [
        r"^why (do|did|does|is|are|would|should)",
        r"^what if ",
        r"^did you know",
        r"^have you ever",
        r"^how (many|much|often) ",
    ]
}

# Explanatory patterns (NOT contrarian)
EXPLANATORY_PATTERNS = [
    r"(why|how|where|what|reasons?) .* (we|you|they|people) .* wrong",
    r"understanding .* wrong",
    r"things .* wrong",
    r"(one of|some of) .* wrong",
]

# Tier 2 Patterns (Strong Amplifiers)
TIER2_PATTERNS = {
    "imperative": [
        r"^stop (doing|saying|making|trying|believing|scrolling)",
        r"^never (do|say|tell|make|try|believe)",
        r"^don't (do|say|make|try|believe|miss)",
        r"^listen,? ",
        r"^wait,? ",
        r"^watch (out|this)",
    ],
    "secret_reveal": [
        r"what (they|nobody|no one|people) (don't|won't|didn't|never) (tell|say|want you to)",
        r"(here's|this is) what .* (don't want|won't tell|hide)",
        r"the (secret|insider|hidden) ",
    ]
}

# Weak Question Patterns (Conversational, not hooks)
WEAK_QUESTION_PATTERNS = [
    r"^what (were|was|are|is) you",      # "What were you like?"
    r"^how (are|do|did) you",            # "How are you?"
    r"^(do|does|did) you (get|have)",    # "Do you get more?"
    r", right\?$",                       # "..., right?"
    r"^(can|could|would|should) you",    # "Could you explain?"
    r"^what (about|do|did) ",            # "What about X?" "What do you think?"
]

# Negative Patterns (Disqualifiers)
NEGATIVE_PATTERNS = {
    "mid_conversation": r"^(and |but |so |well |also |one is |because )",
    "filler": r"^(um,? |uh,? |like,? |you know,? |okay,? |i mean,? )",
    "greeting": r"^(hello|hi everyone|welcome|hey guys|good morning)",
    "context_ref": r"(as i (said|mentioned)|like i said|going back|remember when|earlier we)",
    "context_dependent_start": r"^(is that|to know|that you|which you|when you|to feel)",
    "academic": [
        "definition", "framework", "taxonomy", "scholars",
        "according to", "research shows", "experts say",
        "methodology", "analysis", "hypothesis", "theory"
    ],
    "passive_first_person": r"^i (was|am|have been) (thinking|wondering|realizing|noticing|feeling)",
    "vague_subject": r"^(this|that|it) (is|was|will|can) "
}


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ScoredSegment:
    """Segment with hook score and metadata."""
    id: int
    start: float
    end: float
    text: str
    hook_score: int
    signals: List[str]
    reason: str
    tier_breakdown: Dict[str, int]


# ============================================================================
# HOOK SCORING ENGINE V3
# ============================================================================

def score_hook(text: str, segment_id: int) -> Tuple[int, List[str], str, Dict[str, int]]:
    """
    Score segment as hook (1-10) with research-backed patterns.
    
    VERSION 3: Enhanced filtering for incomplete sentences, refined contrarian,
    short statement context checks, and demoted weak questions.
    
    Returns:
        (score, signals, reason, tier_breakdown)
    """
    score = 5
    signals = []
    reasons = []
    tier_breakdown = {"tier1": 0, "tier2": 0, "tier3": 0}
    
    text_lower = text.lower().strip()
    first_15_words = ' '.join(text.split()[:15]).lower()
    next_15_words = ' '.join(text.split()[15:30]).lower()
    first_5_words = ' '.join(text.split()[:5]).lower()
    word_count = len(text.split())
    
    # ========================================
    # INSTANT DISQUALIFIERS
    # ========================================
    
    # Fragment detection (starts with lowercase = mid-sentence)
    if text and text[0].islower():
        return 1, ["rejected_fragment"], "sentence fragment", tier_breakdown
    
    # Mid-conversation start
    if re.search(NEGATIVE_PATTERNS["mid_conversation"], text_lower):
        return 1, ["rejected_mid_conversation"], "mid-conversation start", tier_breakdown
    
    # Context-dependent starts (pronouns without antecedent)
    if re.search(NEGATIVE_PATTERNS["context_dependent_start"], text_lower):
        return 1, ["rejected_context_dependent"], "needs prior context", tier_breakdown
    
    # Greeting/filler
    if re.search(NEGATIVE_PATTERNS["greeting"], text_lower):
        return 1, ["rejected_greeting"], "greeting intro", tier_breakdown
    
    if re.search(NEGATIVE_PATTERNS["filler"], text_lower):
        score -= 3
        signals.append("filler_words")
        reasons.append("filler")
    
    # FIX 1: Incomplete sentence detection
    incomplete_indicators = [r",$"]  # Ends with comma
    has_ending_punctuation = text.rstrip().endswith(('.', '!', '?'))
    
    if word_count >= 5 and not has_ending_punctuation:
        score -= 2
        signals.append("incomplete_sentence")
        reasons.append("incomplete")
    
    # FIX 3: Context-dependent short statements
    if word_count <= 5 and has_ending_punctuation:
        # Check if it's self-explanatory
        standalone_short = [
            r"^stop ",
            r"^never ",
            r"^why ",
            r"^what if",
            r"^did you know",
        ]
        
        if not any(re.search(p, text_lower) for p in standalone_short):
            score -= 2
            signals.append("too_short_needs_context")
            reasons.append("needs context")
    
    # ========================================
    # TIER 1: SCROLL-STOPPERS (+3 each)
    # ========================================
    
    tier1_matches = []
    
    # 1. Open Loops
    for pattern in TIER1_PATTERNS["open_loop"]:
        if re.search(pattern, text_lower):
            tier1_matches.append("open_loop")
            signals.append("open_loop")
            reasons.append("curiosity gap")
            break
    
    # 2. Contrarian Language (REFINED - FIX 2)
    is_contrarian_assertion = any(
        re.search(p, text_lower) for p in TIER1_PATTERNS["contrarian_assertion"]
    )
    is_explanatory = any(
        re.search(p, text_lower) for p in EXPLANATORY_PATTERNS
    )
    
    if is_contrarian_assertion and not is_explanatory:
        tier1_matches.append("contrarian")
        signals.append("contrarian")
        reasons.append("challenges belief")
    elif "wrong" in text_lower and not is_explanatory:
        # Mild contrarian signal (Tier 3)
        score += 1
        tier_breakdown["tier3"] += 1
        signals.append("mild_contrarian")
    
    # 3. Question Hook (REFINED - Strong vs Weak) - FIX 4
    if '?' in text:
        # Check if it's a STRONG question (provocative)
        is_strong_question = any(
            re.search(p, text_lower) for p in TIER1_PATTERNS["strong_question"]
        )
        
        # Check if it's a WEAK question (conversational)
        is_weak_question = any(
            re.search(p, text_lower) for p in WEAK_QUESTION_PATTERNS
        )
        
        if is_strong_question:
            tier1_matches.append("strong_question")
            signals.append("strong_question")
            reasons.append("provocative question")
        elif is_weak_question:
            # Weak question = NO BONUS (FIX 4)
            signals.append("weak_question")
        else:
            # Generic question = NO BONUS (FIX 4)
            signals.append("generic_question")
    
    # 4. Strong Direct Address
    for pattern in TIER1_PATTERNS["strong_address"]:
        if re.search(pattern, first_15_words):
            tier1_matches.append("strong_address")
            signals.append("strong_address")
            reasons.append("commands attention")
            break
    
    # Apply Tier 1 scoring with soft cap
    tier1_count = len(tier1_matches)
    if tier1_count >= 3:
        # Multiple distinct patterns = exceptional (full credit)
        tier1_contribution = tier1_count * 3
    else:
        # Soft cap at +5 for typical hooks
        tier1_contribution = min(tier1_count * 3, TIER1_SOFT_CAP)
    
    score += tier1_contribution
    tier_breakdown["tier1"] = tier1_contribution
    
    # ========================================
    # TIER 2: STRONG AMPLIFIERS (+2 each)
    # ========================================
    
    # 5. Imperatives
    for pattern in TIER2_PATTERNS["imperative"]:
        if re.search(pattern, text_lower):
            score += 2
            tier_breakdown["tier2"] += 2
            signals.append("imperative")
            reasons.append("urgent command")
            break
    
    # 6. Power Words - Early Position
    power_found = False
    for category, words in POWER_WORDS.items():
        found_early = [w for w in words if w in first_15_words]
        if found_early:
            score += 2
            tier_breakdown["tier2"] += 2
            signals.append(f"power_{category}")
            reasons.append(f"{category}: {found_early[0]}")
            power_found = True
            break
    
    # 7. Specificity with Numbers
    if re.search(r'(\$|€|£)?\d+[%KkMm]?|\d+\s*(percent|years|months|times|ways|things)', text):
        score += 2
        tier_breakdown["tier2"] += 2
        signals.append("specific_numbers")
        reasons.append("concrete data")
    
    # 8. Secret/Insider Reveal
    for pattern in TIER2_PATTERNS["secret_reveal"]:
        if re.search(pattern, text_lower):
            score += 2
            tier_breakdown["tier2"] += 2
            signals.append("secret_reveal")
            reasons.append("insider knowledge")
            break
    
    # TIER 2 PENALTIES
    
    # 9. Vague Subjects
    if re.search(NEGATIVE_PATTERNS["vague_subject"], text_lower):
        # Check if rescued by Tier 1/2 hook
        has_rescue = tier1_count > 0 or any(
            s in signals for s in ['imperative', 'secret_reveal']
        )
        if not has_rescue:
            score -= 2
            tier_breakdown["tier2"] -= 2
            signals.append("vague_subject")
            reasons.append("unclear reference")
    
    # 10. Academic Language
    if any(kw in text_lower for kw in NEGATIVE_PATTERNS["academic"]):
        score -= 2
        tier_breakdown["tier2"] -= 2
        signals.append("academic")
        reasons.append("academic tone")
    
    # 11. Passive First-Person
    if re.search(NEGATIVE_PATTERNS["passive_first_person"], text_lower):
        score -= 2
        tier_breakdown["tier2"] -= 2
        signals.append("passive_reflection")
        reasons.append("passive intro")
    
    # 12. Conversational Tag Question Penalty
    if text_lower.endswith(', right?') or text_lower.endswith('right?'):
        score -= 2
        tier_breakdown["tier2"] -= 2
        signals.append("conversational_tag")
        reasons.append("tag question")
    
    # ========================================
    # TIER 3: MINOR SIGNALS (+1 each)
    # ========================================
    
    # 13. Weak Direct Address
    if "strong_address" not in signals:
        if any(w in first_5_words for w in ['you', 'your', "you're", "you've"]):
            score += 1
            tier_breakdown["tier3"] += 1
            signals.append("weak_address")
    
    # 14. Dramatic First-Person (only if no strong hooks)
    if tier1_count == 0 and "imperative" not in signals:
        if re.search(r"^i (was|hit|lost|went|got) .*(debt|broke|bottom|fired|wrong)", text_lower):
            score += 1
            tier_breakdown["tier3"] += 1
            signals.append("story_setup")
    
    # 15. Emotional Polarity Words
    emotional = ["shocking", "insane", "unbelievable", "incredible",
                "terrifying", "amazing", "mind-blowing", "jaw-dropping"]
    if any(word in first_15_words for word in emotional):
        score += 1
        tier_breakdown["tier3"] += 1
        signals.append("emotional_amplifier")
    
    # 16. Power Words - Mid Position (if not already scored)
    if not power_found:
        for category, words in POWER_WORDS.items():
            found_mid = [w for w in words if w in next_15_words]
            if found_mid:
                score += 1
                tier_breakdown["tier3"] += 1
                signals.append(f"power_{category}_mid")
                break
    
    # 17. Brevity Bonus
    if word_count <= 15:
        score += 1
        tier_breakdown["tier3"] += 1
        signals.append("concise")
    
    # TIER 3 PENALTIES
    
    # 18. Verbosity
    if word_count > 30:
        score -= 1
        tier_breakdown["tier3"] -= 1
        signals.append("verbose")
    
    # ========================================
    # FINALIZE
    # ========================================
    
    # Clamp score
    score = max(1, min(10, score))
    
    # Build reason string
    reason_str = '; '.join(reasons) if reasons else 'neutral'
    
    return score, signals, reason_str, tier_breakdown


# ============================================================================
# CANDIDATE DISCOVERY
# ============================================================================

def discover_hooks(
    transcript_path: str,
    min_score: int = MIN_HOOK_SCORE
) -> List[ScoredSegment]:
    """
    Scan all transcript segments and score for hook quality.
    
    Returns ranked list of high-scoring candidates.
    """
    # Load transcript
    with open(transcript_path, 'r') as f:
        data = json.load(f)
    
    segments = data.get("segments", data) if isinstance(data, dict) else data
    
    print(f"\n{'='*70}")
    print("HOOK DISCOVERY ENGINE V3 (Production Release)")
    print(f"{'='*70}")
    print(f"Scanning {len(segments)} segments...")
    print(f"Min score threshold: {min_score}/10")
    print(f"Tier 1 soft cap: {TIER1_SOFT_CAP} points (unless 3+ distinct patterns)")
    print()
    print("V3 ENHANCEMENTS:")
    print("  • Incomplete sentence detection & penalty")
    print("  • Refined contrarian (assertions vs explanatory)")
    print("  • Short statement context dependency check")
    print("  • Generic/weak questions demoted to neutral")
    print()
    
    candidates = []
    score_distribution = {i: 0 for i in range(1, 11)}
    rejected_count = {
        "fragment": 0,
        "mid_conversation": 0,
        "context_dependent": 0,
        "greeting": 0
    }
    
    for seg in segments:
        text = seg.get("text", "").strip()
        
        # Skip empty or very short
        if not text or len(text) < 15:
            continue
        
        seg_id = seg.get("id", seg.get("idx", 0))
        
        # Score the hook
        score, signals, reason, tier_breakdown = score_hook(text, seg_id)
        
        # Track rejections
        if score == 1:
            for reject_type in rejected_count.keys():
                if f"rejected_{reject_type}" in signals:
                    rejected_count[reject_type] += 1
        
        # Track distribution
        score_distribution[score] += 1
        
        # Only keep high-scoring
        if score >= min_score:
            candidates.append(ScoredSegment(
                id=seg_id,
                start=seg["start"],
                end=seg["end"],
                text=text,
                hook_score=score,
                signals=signals,
                reason=reason,
                tier_breakdown=tier_breakdown
            ))
    
    # Sort by score (descending)
    candidates.sort(key=lambda x: x.hook_score, reverse=True)
    
    # Print rejection stats
    print(f"{'─'*70}")
    print("REJECTION STATS:")
    print(f"{'─'*70}")
    total_rejected = sum(rejected_count.values())
    for reject_type, count in rejected_count.items():
        if count > 0:
            print(f"  {reject_type.replace('_', ' ').title()}: {count}")
    print(f"  Total Rejected: {total_rejected}")
    print()
    
    # Print statistics
    print(f"{'─'*70}")
    print("SCORE DISTRIBUTION:")
    print(f"{'─'*70}")
    for score in range(10, 0, -1):
        count = score_distribution[score]
        bar = '█' * min(count, 50)
        print(f"  {score:2d}/10: {bar} ({count})")
    print()
    
    print(f"{'─'*70}")
    print(f"✓ Found {len(candidates)} candidates with score ≥{min_score}")
    print(f"{'─'*70}\n")
    
    # Show top 15
    print("TOP 15 HOOK CANDIDATES:")
    print(f"{'─'*70}")
    for i, seg in enumerate(candidates[:15], 1):
        print(f"{i:2d}. Score: {seg.hook_score}/10 | {seg.start:.1f}s")
        print(f"    Tier: T1={seg.tier_breakdown['tier1']:+d} "
              f"T2={seg.tier_breakdown['tier2']:+d} "
              f"T3={seg.tier_breakdown['tier3']:+d}")
        print(f"    Signals: {', '.join(seg.signals[:5])}")
        print(f"    Text: {seg.text[:90]}...")
        print()
    
    return candidates


def save_candidates(candidates: List[ScoredSegment], output_path: str):
    """Save candidates to JSON for LLM refinement."""
    
    output_data = {
        "metadata": {
            "total_candidates": len(candidates),
            "min_score": min(c.hook_score for c in candidates) if candidates else 0,
            "max_score": max(c.hook_score for c in candidates) if candidates else 0,
            "avg_score": round(sum(c.hook_score for c in candidates) / len(candidates), 1) if candidates else 0,
            "score_10_count": sum(1 for c in candidates if c.hook_score == 10),
            "score_9_count": sum(1 for c in candidates if c.hook_score == 9),
            "score_8_count": sum(1 for c in candidates if c.hook_score == 8),
        },
        "candidates": [asdict(c) for c in candidates[:MAX_CANDIDATES]]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"{'='*70}")
    print(f"✓ Saved top {min(len(candidates), MAX_CANDIDATES)} candidates to:")
    print(f"  {output_path}")
    print(f"\nSTATS:")
    print(f"  Score 10: {output_data['metadata']['score_10_count']} clips")
    print(f"  Score 9:  {output_data['metadata']['score_9_count']} clips")
    print(f"  Score 8:  {output_data['metadata']['score_8_count']} clips")
    print(f"  Average:  {output_data['metadata']['avg_score']}/10")
    print(f"{'='*70}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Validate inputs
    if not Path(TRANSCRIPT_PATH).exists():
        print(f"❌ ERROR: Transcript not found: {TRANSCRIPT_PATH}")
        print("   Run transcription first")
        return 1
    
    # Discover hooks
    candidates = discover_hooks(TRANSCRIPT_PATH, min_score=MIN_HOOK_SCORE)
    
    if not candidates:
        print("\n⚠️  No candidates found above threshold")
        print(f"   Try lowering MIN_HOOK_SCORE (currently {MIN_HOOK_SCORE})")
        print(f"   Or check if transcript has strong hook patterns")
        return 1
    
    # Save for LLM refinement
    save_candidates(candidates, OUTPUT_PATH)
    
    print(f"\n{'='*70}")
    print("✅ HOOK DISCOVERY COMPLETE")
    print(f"   Next step: Run LLM refinement to select final clips")
    print(f"   Command: python scripts/orchestrate.py --input <video>")
    print(f"{'='*70}\n")
    
    return 0

 
if __name__ == "__main__":
    sys.exit(main())
