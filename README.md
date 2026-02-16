# RO Growth Engine

End-to-end system for podcast short-form content: automated clipping pipeline + performance analytics + ML-based engagement prediction. Identifies what makes clips go viral and uses those signals to improve future clip selection.

**Why this exists:** Clipping is only half the problem. The other half is knowing *which* clips will actually perform. This system closes the feedback loop — analyze what worked, build a model, use it to score future clips before posting.

## System overview

Two parallel pipelines that feed into each other:

```
                    ┌─────────────────────────────┐
                    │       CLIP PIPELINE          │
                    │  Video → Hooks → LLM → Clips │
                    └──────────────┬──────────────┘
                                   │
                          rendered clips posted
                                   │
                    ┌──────────────v──────────────┐
                    │     ANALYSIS PIPELINE        │
                    │  YouTube API → Transcribe →   │
                    │  Tag → ML → Predict           │
                    └──────────────┬──────────────┘
                                   │
                      engagement signals feed back
                      into hook scoring weights
```

### Clip pipeline (9 stages)

Transforms a podcast episode into ready-to-post short-form clips:

1. **Download** — yt-dlp pulls video from YouTube
2. **Extract audio** — FFmpeg converts to MP3
3. **Transcribe** — Whisper generates word-level timestamps
4. **Discover hooks** — Python scoring engine (V3) identifies 30 candidates
5. **Generate context** — Gemini 2.5 Flash analyzes full episode (1M token context)
6. **Select clips** — LLM refines candidates, validates hook + payoff
7. **Resolve timestamps** — Snap to sentence boundaries
8. **Render** — FFmpeg batch renders MP4 clips
9. **Subtitles** — Generate SRT from word-level transcript

### Analysis pipeline (6 stages)

Analyzes existing Shorts performance to train the engagement model:

1. **Extract performance** — YouTube Data API pulls views, likes, comments
2. **Normalize metrics** — Compute channel-relative performance indices (VPI, LPI, CPI)
3. **Download audio** — Fetch Shorts audio for transcription
4. **Transcribe + tag** — Quantitative features (WPM, duration, filler density) + qualitative tags (hook type, topic, emotion) via Groq
5. **Load to SQLite** — Structured storage for ML training
6. **Train model** — CatBoost classifier predicts top-25% engagement

### Engagement model

**Target:** Binary classification — will this clip land in the top 25% of engagement for this channel?

**Features (20):**

| Type | Features |
|------|----------|
| Speech | duration, word count, WPM, hook WPM, filler count/density |
| Structure | sentence count, reading level, question opener, first/second person ratio |
| Semantic | hook type, hook emotion, primary topic, technical depth |
| Content | has examples, has payoff, has numbers, insider language |

**Approach:** CatBoost with SHAP interpretability — not just predictions, but *why* each clip is predicted to perform. Feature importance drives improvements to the hook scoring algorithm.

**Normalization:** All metrics are channel-relative (VPI = views / channel mean), so the model generalizes across channels with different audience sizes.

## Tech stack

- **Python 3.13**, FFmpeg, yt-dlp
- **Gemini 2.5 Flash** — episode analysis and clip selection
- **Groq** (Llama 3.3-70b) — qualitative tagging
- **Whisper** — transcription
- **CatBoost** — engagement prediction with SHAP explainability
- **SQLite** + SQLAlchemy — structured analytics storage
- **YouTube Data API v3** — performance metrics extraction

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Required environment variables:
```bash
export GOOGLE_API_KEY="your-gemini-key"
export GROQ_API_KEY="your-groq-key"
export YOUTUBE_API_KEY="your-youtube-key"   # for analysis pipeline
```

## Usage

```bash
# Clip pipeline: video → rendered clips
python scripts/clip_pipeline/orchestrate.py \
  --input episode.mp4 \
  --output-dir data/channels/my-podcast/episodes/ep1

# Analysis pipeline: analyze existing Shorts
python scripts/analysis_pipeline/analysis_orchestrate.py \
  --channel "ChannelName" \
  --api-key $YOUTUBE_API_KEY

# Train engagement model
python scripts/analysis_pipeline/engagement_model.py \
  --db data/tagging/clips.db
```

## Project structure

```
scripts/
├── clip_pipeline/           # Video → clips
│   ├── orchestrate.py       # 9-stage pipeline runner
│   ├── discover_hooks.py    # Hook scoring algorithm (V3)
│   ├── llm_client.py        # Gemini 2.5 Flash integration
│   └── groq_api_client.py   # Groq fallback
│
├── analysis_pipeline/       # Shorts performance analysis
│   ├── analysis_orchestrate.py
│   ├── youtube_performance_extractor.py
│   └── engagement_model.py  # CatBoost + SHAP
│
├── data_pipeline/           # Transcription + tagging
│   ├── pipeline.py
│   └── groq_qual_client.py
│
└── upload_pipeline/         # YouTube upload (WIP)
```
