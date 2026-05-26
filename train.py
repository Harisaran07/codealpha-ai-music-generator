"""
train.py — Train the LSTM model on preprocessed music data.

Handles loading cached data, building the model, and running
the training loop with callbacks for checkpointing and early stopping.
"""

import os
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

import config
from utils import load_pickle, print_banner, print_step
from preprocess import preprocess_data, create_sequences
from model import build_model, print_model_summary


def load_training_data():
    """
    Load preprocessed data from cache, or run preprocessing if needed.

    Returns:
        tuple: (network_input, network_output, n_vocab)
    """
    print_step(1, "Loading training data")

    # Check if cached data exists
    if os.path.exists(config.NOTES_FILE) and os.path.exists(config.VOCAB_FILE):
        print("  Found cached data, loading...")

        notes = load_pickle(config.NOTES_FILE)
        vocab_data = load_pickle(config.VOCAB_FILE)

        n_vocab = vocab_data["n_vocab"]
        note_to_int = vocab_data["note_to_int"]

        # Recreate sequences from cached notes
        network_input, network_output, _, _, _ = create_sequences(notes)

        return network_input, network_output, n_vocab
    else:
        print("  No cached data found, running preprocessing...")
        result = preprocess_data()
        if result is None:
            return None, None, None
        network_input, network_output, n_vocab, _, _ = result
        return network_input, network_output, n_vocab


def train_model():
    """
    Main training function: load data → build model → train → save.
    """
    print_banner("Model Training")
    config.ensure_dirs()

    # Load data
    network_input, network_output, n_vocab = load_training_data()

    if network_input is None:
        print("  ✗ Failed to load training data. Aborting.")
        return None

    # Build model
    print_step(2, "Building LSTM model")
    model = build_model(network_input.shape, n_vocab)
    print_model_summary(model)

    # Check for existing weights to resume training
    if os.path.exists(config.MODEL_CHECKPOINT):
        print(f"\n  Found existing weights: {config.MODEL_CHECKPOINT}")
        print("  Loading weights to resume training...")
        try:
            model.load_weights(config.MODEL_CHECKPOINT)
            print("  ✓ Weights loaded successfully")
        except Exception as e:
            print(f"  ⚠ Could not load weights: {e}")
            print("  Starting fresh training...")

    # Define callbacks
    callbacks = [
        # Save best model based on lowest loss
        ModelCheckpoint(
            config.MODEL_CHECKPOINT,
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # Stop training if no improvement
        EarlyStopping(
            monitor="loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            verbose=1,
            restore_best_weights=True,
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor="loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=1,
            min_lr=1e-6,
        ),
    ]

    # Train!
    print_step(3, "Training")
    print(f"  Epochs:     {config.EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Samples:    {network_input.shape[0]}")
    print()

    history = model.fit(
        network_input,
        network_output,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Training summary
    final_loss = history.history["loss"][-1]
    best_loss = min(history.history["loss"])
    total_epochs = len(history.history["loss"])

    print(f"\n{'═' * 50}")
    print(f"  Training Complete!")
    print(f"  Total epochs trained: {total_epochs}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Best loss:  {best_loss:.4f}")
    print(f"  Model saved to: {config.MODEL_CHECKPOINT}")
    print(f"{'═' * 50}")

    return model, history


if __name__ == "__main__":
    train_model()
