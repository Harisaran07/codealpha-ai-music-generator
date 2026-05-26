"""
generate.py — Generate new music using the trained LSTM model.

Loads a trained model, picks a seed sequence, and iteratively
predicts new notes using temperature sampling. The generated
sequence is then converted to a MIDI file.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from music21 import instrument, note, chord, stream

import config
from utils import load_pickle, sample_with_temperature, print_banner, print_step


def load_generation_resources():
    """
    Load the trained model and vocabulary mappings.

    Returns:
        tuple: (model, notes, vocab_data) or (None, None, None) on failure
    """
    print_step(1, "Loading model and vocabulary")

    # Check required files exist
    if not os.path.exists(config.MODEL_CHECKPOINT):
        print(f"  ✗ Model not found: {config.MODEL_CHECKPOINT}")
        print("  Please train the model first: python main.py --train")
        return None, None, None

    if not os.path.exists(config.NOTES_FILE) or not os.path.exists(config.VOCAB_FILE):
        print("  ✗ Preprocessed data not found.")
        print("  Please preprocess first: python main.py --preprocess")
        return None, None, None

    # Load model
    print("  Loading trained model...")
    model = load_model(config.MODEL_CHECKPOINT)
    print("  ✓ Model loaded")

    # Load notes and vocabulary
    notes = load_pickle(config.NOTES_FILE)
    vocab_data = load_pickle(config.VOCAB_FILE)

    print(f"  ✓ Vocabulary size: {vocab_data['n_vocab']}")

    return model, notes, vocab_data


def generate_notes(model, notes, vocab_data, length=None, temperature=None):
    """
    Generate a sequence of new notes using the trained model.

    Args:
        model: trained Keras model
        notes: list of str — original note sequences (for seed selection)
        vocab_data: dict with 'note_to_int', 'int_to_note', 'n_vocab'
        length: int — number of notes to generate (default from config)
        temperature: float — sampling temperature (default from config)

    Returns:
        list of str — generated note/chord strings
    """
    if length is None:
        length = config.GENERATION_LENGTH
    if temperature is None:
        temperature = config.TEMPERATURE

    note_to_int = vocab_data["note_to_int"]
    int_to_note = vocab_data["int_to_note"]
    n_vocab = vocab_data["n_vocab"]

    print_step(2, "Generating music")
    print(f"  Notes to generate: {length}")
    print(f"  Temperature: {temperature}")

    # Pick a random starting point from the training data
    start_idx = np.random.randint(0, len(notes) - config.SEQUENCE_LENGTH)
    pattern = [note_to_int[n] for n in notes[start_idx : start_idx + config.SEQUENCE_LENGTH]]

    print(f"  Seed sequence from index: {start_idx}")

    generated_notes = []

    for i in range(length):
        # Prepare input
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        # Predict next note probabilities
        prediction = model.predict(prediction_input, verbose=0)

        # Sample from the distribution using temperature
        index = sample_with_temperature(prediction[0], temperature)

        # Convert back to note string
        result = int_to_note[index]
        generated_notes.append(result)

        # Slide the window: remove first, add predicted
        pattern.append(index)
        pattern = pattern[1:]

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{length} notes...")

    print(f"  ✓ Generated {len(generated_notes)} notes")
    return generated_notes


def notes_to_midi(generated_notes, output_file=None):
    """
    Convert generated note strings back to a MIDI file.

    Args:
        generated_notes: list of str — note/chord strings
        output_file: str — output .mid file path (default from config)

    Returns:
        str — path to the saved MIDI file
    """
    if output_file is None:
        output_file = config.OUTPUT_MIDI_FILE

    print_step(3, "Converting to MIDI")

    offset = 0  # Current time position
    output_notes = []

    for pattern in generated_notes:
        try:
            # Check if it's a chord (contains dots between numbers)
            if ("." in pattern) or pattern.isdigit():
                # It's a chord — parse individual notes
                notes_in_chord = pattern.split(".")
                chord_notes = []
                for current_note in notes_in_chord:
                    try:
                        new_note = note.Note(int(current_note))
                        new_note.storedInstrument = instrument.Piano()
                        chord_notes.append(new_note)
                    except (ValueError, Exception):
                        pass

                if chord_notes:
                    new_chord = chord.Chord(chord_notes)
                    new_chord.offset = offset
                    output_notes.append(new_chord)
            else:
                # It's a single note
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

        except Exception as e:
            # Skip malformed patterns
            pass

        # Advance time by 0.5 beats (eighth note spacing)
        offset += 0.5

    # Create the MIDI stream
    midi_stream = stream.Stream(output_notes)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write to MIDI file
    midi_stream.write("midi", fp=output_file)

    print(f"  ✓ MIDI file saved: {output_file}")
    print(f"  ✓ Total notes/chords in output: {len(output_notes)}")
    print(f"  ✓ Duration: ~{offset / 2:.0f} seconds (at 120 BPM)")

    return output_file


def generate_music(temperature=None, length=None):
    """
    Full generation pipeline: load model → generate → save MIDI.

    Args:
        temperature: float — sampling temperature (optional)
        length: int — number of notes to generate (optional)

    Returns:
        str — path to generated MIDI file, or None on failure
    """
    print_banner("Music Generation")
    config.ensure_dirs()

    # Load resources
    model, notes, vocab_data = load_generation_resources()
    if model is None:
        return None

    # Generate notes
    generated_notes = generate_notes(
        model, notes, vocab_data,
        length=length,
        temperature=temperature,
    )

    # Convert to MIDI
    output_file = notes_to_midi(generated_notes)

    print(f"\n{'═' * 50}")
    print(f"  🎵 Music Generation Complete!")
    print(f"  Output: {output_file}")
    print(f"  Open this .mid file in any MIDI player to listen!")
    print(f"{'═' * 50}")

    return output_file


if __name__ == "__main__":
    generate_music()
