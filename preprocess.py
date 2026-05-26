"""
preprocess.py — Parse MIDI files and prepare training data.

Uses music21 to extract notes and chords from MIDI files,
then creates integer-encoded sequences for LSTM training.
"""

import os
import glob
import numpy as np
from music21 import converter, instrument, note, chord

import config
from utils import save_pickle, load_pickle, print_banner, print_step


def parse_midi_file(filepath):
    """
    Parse a single MIDI file and extract notes and chords.

    Args:
        filepath: str — path to .mid file

    Returns:
        list of str — note/chord representations (e.g., "C4", "C4.E4.G4")
    """
    notes = []

    try:
        midi = converter.parse(filepath)

        # Try to get parts; if multi-instrument, flatten
        try:
            parts = instrument.partitionByInstrument(midi)
            if parts:
                # Use the first part (usually the melody/right hand)
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
        except Exception:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                # Single note: store as pitch string (e.g., "C4")
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                # Chord: store as dot-separated pitch strings (e.g., "C4.E4.G4")
                notes.append(".".join(str(n) for n in element.normalOrder))

    except Exception as e:
        print(f"  ⚠ Error parsing {os.path.basename(filepath)}: {e}")

    return notes


def extract_all_notes():
    """
    Extract notes from all MIDI files in the data directory.

    Returns:
        list of str — all extracted notes/chords in sequence
    """
    print_step(1, "Extracting notes from MIDI files")

    all_notes = []
    midi_files = glob.glob(os.path.join(config.MIDI_DATA_DIR, "**", "*.mid"), recursive=True)
    midi_files += glob.glob(os.path.join(config.MIDI_DATA_DIR, "**", "*.midi"), recursive=True)

    # Remove duplicates (in case .mid and .midi overlap)
    midi_files = list(set(midi_files))

    if not midi_files:
        print(f"  ⚠ No MIDI files found in {config.MIDI_DATA_DIR}")
        print("  Please run data collection first: python main.py --download")
        return []

    print(f"  Found {len(midi_files)} MIDI files")

    for i, filepath in enumerate(sorted(midi_files)):
        filename = os.path.basename(filepath)
        print(f"  [{i + 1}/{len(midi_files)}] Parsing: {filename}", end="")

        notes = parse_midi_file(filepath)
        all_notes.extend(notes)
        print(f" → {len(notes)} notes")

    print(f"\n  Total notes extracted: {len(all_notes)}")
    return all_notes


def create_sequences(notes):
    """
    Create input-output sequence pairs for LSTM training.

    Args:
        notes: list of str — extracted note/chord strings

    Returns:
        tuple: (network_input, network_output, n_vocab, note_to_int, int_to_note)
            - network_input: numpy array of shape (n_samples, sequence_length, 1), normalized
            - network_output: numpy array of shape (n_samples, n_vocab), one-hot encoded
            - n_vocab: int — number of unique notes/chords
            - note_to_int: dict — mapping from note string to integer
            - int_to_note: dict — mapping from integer to note string
    """
    print_step(2, "Creating training sequences")

    # Build vocabulary
    unique_notes = sorted(set(notes))
    n_vocab = len(unique_notes)
    print(f"  Unique notes/chords: {n_vocab}")

    # Create mappings
    note_to_int = {note: i for i, note in enumerate(unique_notes)}
    int_to_note = {i: note for i, note in enumerate(unique_notes)}

    # Create input-output pairs using sliding window
    network_input = []
    network_output = []

    for i in range(len(notes) - config.SEQUENCE_LENGTH):
        # Input: sequence of SEQUENCE_LENGTH notes
        seq_in = [note_to_int[n] for n in notes[i : i + config.SEQUENCE_LENGTH]]
        # Output: the next note after the sequence
        seq_out = note_to_int[notes[i + config.SEQUENCE_LENGTH]]

        network_input.append(seq_in)
        network_output.append(seq_out)

    n_patterns = len(network_input)
    print(f"  Training patterns created: {n_patterns}")

    # Reshape input for LSTM: (n_samples, sequence_length, 1)
    network_input = np.reshape(network_input, (n_patterns, config.SEQUENCE_LENGTH, 1))

    # Normalize input to [0, 1]
    network_input = network_input / float(n_vocab)

    # One-hot encode output
    network_output = np.eye(n_vocab)[network_output]

    print(f"  Input shape:  {network_input.shape}")
    print(f"  Output shape: {network_output.shape}")

    return network_input, network_output, n_vocab, note_to_int, int_to_note


def preprocess_data():
    """
    Main preprocessing pipeline: extract notes → create sequences → save to disk.

    Returns:
        tuple: (network_input, network_output, n_vocab, note_to_int, int_to_note)
    """
    print_banner("Data Preprocessing")
    config.ensure_dirs()

    # Step 1: Extract notes
    notes = extract_all_notes()

    if not notes:
        print("  ✗ No notes extracted. Cannot proceed.")
        return None

    if len(notes) < config.SEQUENCE_LENGTH + 1:
        print(f"  ✗ Not enough notes ({len(notes)}). Need at least {config.SEQUENCE_LENGTH + 1}.")
        return None

    # Cache raw notes
    save_pickle(notes, config.NOTES_FILE)

    # Step 2: Create sequences
    network_input, network_output, n_vocab, note_to_int, int_to_note = create_sequences(notes)

    # Cache vocabulary mappings
    vocab_data = {
        "note_to_int": note_to_int,
        "int_to_note": int_to_note,
        "n_vocab": n_vocab,
    }
    save_pickle(vocab_data, config.VOCAB_FILE)

    print(f"\n  ✓ Preprocessing complete!")
    print(f"  ✓ Notes cached to: {config.NOTES_FILE}")
    print(f"  ✓ Vocab cached to: {config.VOCAB_FILE}")

    return network_input, network_output, n_vocab, note_to_int, int_to_note


if __name__ == "__main__":
    preprocess_data()
