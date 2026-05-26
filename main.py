"""
main.py — End-to-end orchestrator for the AI Music Generation project.

Provides a single CLI entry point to run any or all steps:
    python main.py --download     Download MIDI training data
    python main.py --preprocess   Preprocess MIDI into sequences
    python main.py --train        Train the LSTM model
    python main.py --generate     Generate new music
    python main.py --all          Run the full pipeline

Optional flags:
    --epochs N          Override number of training epochs
    --temperature F     Override generation temperature (0.1 - 2.0)
    --length N          Override number of notes to generate
"""

import argparse
import sys
import os

# Suppress TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import config
from utils import print_banner


def main():
    parser = argparse.ArgumentParser(
        description="AI Music Generator — LSTM-based music generation from MIDI data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                      Run full pipeline
  python main.py --download --preprocess    Download and preprocess only
  python main.py --train --epochs 50        Train with 50 epochs
  python main.py --generate --temperature 0.8 --length 200
        """,
    )

    parser.add_argument("--download", action="store_true", help="Download MIDI training data")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess MIDI files into training sequences")
    parser.add_argument("--train", action="store_true", help="Train the LSTM model")
    parser.add_argument("--generate", action="store_true", help="Generate new music from trained model")
    parser.add_argument("--all", action="store_true", help="Run the complete pipeline (download → preprocess → train → generate)")

    # Override parameters
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature (0.1-2.0)")
    parser.add_argument("--length", type=int, default=None, help="Number of notes to generate")

    args = parser.parse_args()

    # If no arguments, show help
    if not any([args.download, args.preprocess, args.train, args.generate, args.all]):
        parser.print_help()
        print("\n  ⚠ No action specified. Use --all to run the complete pipeline.")
        sys.exit(1)

    # Apply overrides
    if args.epochs:
        config.EPOCHS = args.epochs
        print(f"  Override: epochs = {args.epochs}")
    if args.temperature:
        config.TEMPERATURE = args.temperature
        print(f"  Override: temperature = {args.temperature}")
    if args.length:
        config.GENERATION_LENGTH = args.length
        print(f"  Override: length = {args.length}")

    print_banner("🎵 AI Music Generator")
    print(f"  Project directory: {config.BASE_DIR}")

    # ── Step 1: Download ───────────────────────────────────────
    if args.download or args.all:
        from collect_data import collect_midi_data
        total = collect_midi_data()
        if total == 0 and (args.preprocess or args.all):
            print("\n  ✗ No MIDI files available. Cannot continue.")
            print("  Please manually place .mid files in:", config.MIDI_DATA_DIR)
            sys.exit(1)

    # ── Step 2: Preprocess ─────────────────────────────────────
    if args.preprocess or args.all:
        from preprocess import preprocess_data
        result = preprocess_data()
        if result is None and (args.train or args.all):
            print("\n  ✗ Preprocessing failed. Cannot continue.")
            sys.exit(1)

    # ── Step 3: Train ──────────────────────────────────────────
    if args.train or args.all:
        from train import train_model
        model, history = train_model()
        if model is None:
            print("\n  ✗ Training failed.")
            sys.exit(1)

    # ── Step 4: Generate ───────────────────────────────────────
    if args.generate or args.all:
        from generate import generate_music
        output = generate_music(
            temperature=args.temperature,
            length=args.length,
        )
        if output is None:
            print("\n  ✗ Generation failed.")
            sys.exit(1)

    print_banner("✓ All Done!")
    print("  Thank you for using AI Music Generator 🎶\n")


if __name__ == "__main__":
    main()
