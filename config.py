"""
config.py — Centralized configuration for the Music Generation project.
All hyperparameters, paths, and settings are defined here.
"""

import os

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIDI_DATA_DIR = os.path.join(BASE_DIR, "midi_data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

NOTES_FILE = os.path.join(PROCESSED_DATA_DIR, "notes.pkl")
VOCAB_FILE = os.path.join(PROCESSED_DATA_DIR, "vocab.pkl")
MODEL_CHECKPOINT = os.path.join(MODEL_DIR, "music_gen_best.keras")

# ─── Data Collection ────────────────────────────────────────────────────────────
# piano-midi.de composer pages to scrape
MIDI_SOURCE_URL = "http://www.piano-midi.de/midi_files.htm"
COMPOSERS = ["bach", "beethoven", "chopin", "mozart", "debussy", "schubert"]
MAX_FILES_PER_COMPOSER = 15  # Limit per composer for manageable training

# ─── Preprocessing ──────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 100  # Number of notes in each input sequence

# ─── Model Architecture ────────────────────────────────────────────────────────
LSTM_UNITS_1 = 512
LSTM_UNITS_2 = 512
LSTM_UNITS_3 = 256
DENSE_UNITS = 256
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001

# ─── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ─── Generation ─────────────────────────────────────────────────────────────────
GENERATION_LENGTH = 300      # Number of notes to generate
TEMPERATURE = 1.0            # Sampling temperature (0.5=conservative, 1.0=balanced, 1.5=creative)
OUTPUT_MIDI_FILE = os.path.join(OUTPUT_DIR, "generated_music.mid")


def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    for d in [MIDI_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)
