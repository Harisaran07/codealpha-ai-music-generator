"""
model.py — LSTM model definition for music generation.

Defines a stacked LSTM architecture with dropout and batch normalization
for learning sequential patterns in music data.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
)

import config


def build_model(network_input_shape, n_vocab):
    """
    Build the stacked LSTM model for music generation.

    Architecture:
        LSTM(512) → Dropout → LSTM(512) → Dropout → LSTM(256) → Dropout →
        BatchNorm → Dense(256) → Dropout → Dense(n_vocab, softmax)

    Args:
        network_input_shape: tuple — shape of input data (sequence_length, features)
        n_vocab: int — number of unique notes/chords (output classes)

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()

    # Layer 1: First LSTM layer (returns sequences for stacking)
    model.add(
        LSTM(
            config.LSTM_UNITS_1,
            input_shape=(network_input_shape[1], network_input_shape[2]),
            return_sequences=True,
        )
    )
    model.add(Dropout(config.DROPOUT_RATE))

    # Layer 2: Second LSTM layer (returns sequences for stacking)
    model.add(LSTM(config.LSTM_UNITS_2, return_sequences=True))
    model.add(Dropout(config.DROPOUT_RATE))

    # Layer 3: Third LSTM layer (final, returns single output)
    model.add(LSTM(config.LSTM_UNITS_3))
    model.add(Dropout(config.DROPOUT_RATE))

    # Batch normalization for training stability
    model.add(BatchNormalization())

    # Dense hidden layer
    model.add(Dense(config.DENSE_UNITS))
    model.add(Activation("relu"))
    model.add(Dropout(config.DROPOUT_RATE))

    # Output layer: probability distribution over all possible notes
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))

    # Compile with Adam optimizer
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
    )

    return model


def print_model_summary(model):
    """Print a formatted model summary."""
    print("\n" + "═" * 60)
    print("  Model Architecture")
    print("═" * 60)
    model.summary()
    print("═" * 60)
