-- Core clip registry
CREATE TABLE IF NOT EXISTS clips (
    clip_id TEXT PRIMARY KEY CHECK (clip_id GLOB 'clip[0-9][0-9]*'),
    episode_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Clip metadata (versioning)
CREATE TABLE IF NOT EXISTS clip_meta (
    clip_id TEXT PRIMARY KEY,
    quant_version TEXT NOT NULL DEFAULT 'v1',
    qual_version TEXT NOT NULL DEFAULT 'v1',
    tagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (clip_id) REFERENCES clips(clip_id)
);

-- Quantitative features (Python-computed)
CREATE TABLE IF NOT EXISTS clip_quant (
    clip_id TEXT PRIMARY KEY,
    duration_s REAL NOT NULL,
    word_count INTEGER NOT NULL,
    wpm REAL NOT NULL,
    hook_word_count INTEGER NOT NULL,
    hook_wpm REAL NOT NULL,
    num_sentences INTEGER NOT NULL,
    question_start INTEGER NOT NULL,
    reading_level REAL NOT NULL,
    filler_count INTEGER NOT NULL,
    filler_density REAL NOT NULL,
    first_person_ratio REAL NOT NULL,
    second_person_ratio REAL NOT NULL,
    FOREIGN KEY (clip_id) REFERENCES clips(clip_id)
);

-- Qualitative features (LLM-tagged)
CREATE TABLE IF NOT EXISTS clip_qual (
    clip_id TEXT PRIMARY KEY,
    hook_type TEXT NOT NULL,
    hook_emotion TEXT NOT NULL,
    topic_primary TEXT NOT NULL,
    has_examples INTEGER NOT NULL,
    has_payoff INTEGER NOT NULL,
    has_numbers INTEGER NOT NULL,
    insider_language INTEGER NOT NULL,
    FOREIGN KEY (clip_id) REFERENCES clips(clip_id)
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_episode ON clips(episode_id);
CREATE INDEX IF NOT EXISTS idx_topic ON clip_qual(topic_primary);
CREATE INDEX IF NOT EXISTS idx_hook_type ON clip_qual(hook_type);
CREATE INDEX IF NOT EXISTS idx_hook_emotion ON clip_qual(hook_emotion);
CREATE INDEX IF NOT EXISTS idx_has_payoff ON clip_qual(has_payoff);
CREATE INDEX IF NOT EXISTS idx_has_examples ON clip_qual(has_examples);
CREATE UNIQUE INDEX IF NOT EXISTS idx_clip_episode ON clips(clip_id, episode_id);
