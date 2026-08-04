from __future__ import annotations

PRAGMATIC_LABELS = (
    "implicit_sentiment",
    "sarcasm",
    "irony",
    "idiom_figurative",
    "code_switching",
    "mocking",
)
POLARITY_LABELS = ("negative", "neutral", "positive")
EMOTION_LABELS = ("anger", "disgust", "enjoyment", "fear", "other", "sadness", "surprise")
ALL_LABEL_KEYS = PRAGMATIC_LABELS + ("polarity", "emotion")
DATASET_SPLITS = ("train", "dev", "test")
EXPECTED_SPLIT_COUNTS = {"train": 7998, "dev": 1999, "test": 2000}
TRAINING_SEEDS = (20260521, 20260522, 20260523)
SPLIT_SEED = 20260520
SUBSET_SEED = 20260524
BOOTSTRAP_SEED = 20260525
BOOTSTRAP_RESAMPLES = 1000
MAX_SEQUENCE_LENGTH = 128
RATIONALE_BETA = 0.3
