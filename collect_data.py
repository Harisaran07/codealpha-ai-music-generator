"""
collect_data.py — Download classical piano MIDI files for training.

Downloads MIDI files from piano-midi.de, a curated source of
high-quality classical piano MIDI performances.
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup

import config
from utils import print_banner, print_step


def get_midi_links_for_composer(composer):
    """
    Scrape piano-midi.de for MIDI download links for a given composer.

    Args:
        composer: str — composer name (e.g., "bach", "chopin")

    Returns:
        list of full URLs to .mid files
    """
    url = f"http://www.piano-midi.de/{composer}.htm"
    print(f"  Fetching page: {url}")

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ⚠ Could not fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    midi_links = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".mid") or href.endswith(".midi"):
            # Construct full URL
            if href.startswith("http"):
                full_url = href
            else:
                full_url = f"http://www.piano-midi.de/{href}"
            midi_links.append(full_url)

    return midi_links


def download_file(url, save_path):
    """
    Download a single file from URL to local path.

    Args:
        url: str — URL to download
        save_path: str — local file path to save

    Returns:
        bool — True if successful
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return True
    except requests.RequestException as e:
        print(f"  ⚠ Failed to download {url}: {e}")
        return False


def collect_midi_data():
    """
    Main function: download MIDI files from piano-midi.de for all configured composers.
    """
    print_banner("MIDI Data Collection")
    config.ensure_dirs()

    total_downloaded = 0
    total_skipped = 0

    for composer in config.COMPOSERS:
        print_step("→", f"Composer: {composer.capitalize()}")

        # Create composer subdirectory
        composer_dir = os.path.join(config.MIDI_DATA_DIR, composer)
        os.makedirs(composer_dir, exist_ok=True)

        # Get MIDI links
        links = get_midi_links_for_composer(composer)
        if not links:
            print(f"  No MIDI files found for {composer}")
            continue

        # Limit number of files
        links = links[: config.MAX_FILES_PER_COMPOSER]
        print(f"  Found {len(links)} MIDI files")

        for i, url in enumerate(links):
            # Extract filename from URL
            filename = url.split("/")[-1]
            # Clean filename
            filename = re.sub(r"[^\w\-_\.]", "_", filename)
            save_path = os.path.join(composer_dir, filename)

            # Skip if already downloaded
            if os.path.exists(save_path):
                total_skipped += 1
                continue

            print(f"  [{i + 1}/{len(links)}] Downloading: {filename}")
            if download_file(url, save_path):
                total_downloaded += 1
            else:
                print(f"  ⚠ Skipping {filename}")

            # Be polite to the server
            time.sleep(0.5)

    # Summary
    total_files = count_midi_files()
    print(f"\n{'═' * 50}")
    print(f"  Collection Complete!")
    print(f"  New downloads: {total_downloaded}")
    print(f"  Already had:   {total_skipped}")
    print(f"  Total MIDI files in {config.MIDI_DATA_DIR}: {total_files}")
    print(f"{'═' * 50}")

    if total_files == 0:
        print("\n  ⚠ No MIDI files found! Please either:")
        print("    1. Check your internet connection and re-run")
        print("    2. Manually place .mid files in:", config.MIDI_DATA_DIR)

    return total_files


def count_midi_files():
    """Count total MIDI files in the data directory."""
    count = 0
    for root, dirs, files in os.walk(config.MIDI_DATA_DIR):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                count += 1
    return count


if __name__ == "__main__":
    collect_midi_data()
