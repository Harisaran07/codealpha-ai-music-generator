"""
utils.py — Helper functions used across the project.
"""

import pickle
import os
import sys
import numpy as np

# Ensure UTF-8 output for Windows console to support emojis and box drawing characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

def save_pickle(data, filepath):
    """Save data to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"  ✓ Saved: {filepath}")


def load_pickle(filepath):
    """Load data from a pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"  ✓ Loaded: {filepath}")
    return data


def sample_with_temperature(predictions, temperature=1.0):
    """
    Sample an index from a probability distribution using temperature scaling.

    Args:
        predictions: numpy array of predicted probabilities
        temperature: float controlling randomness
            - < 1.0 = more conservative / repetitive
            - 1.0   = balanced (original distribution)
            - > 1.0 = more creative / diverse

    Returns:
        Sampled index (int)
    """
    predictions = np.asarray(predictions).astype("float64")

    # Apply temperature scaling in log-space
    log_preds = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(log_preds)
    probabilities = exp_preds / np.sum(exp_preds)

    # Sample from the adjusted distribution
    return np.random.choice(len(probabilities), p=probabilities)


def print_banner(text):
    """Print a formatted banner for section headers."""
    width = max(len(text) + 4, 50)
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def print_step(step_num, description):
    """Print a formatted step indicator."""
    print(f"\n{'─' * 40}")
    print(f"  Step {step_num}: {description}")
    print(f"{'─' * 40}")
