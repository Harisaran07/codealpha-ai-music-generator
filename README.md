# CodeAlpha AI Music Generator

A Python-based AI music generation pipeline that downloads MIDI data, preprocesses it, trains a model, and generates new music.

## Repository Structure
- `app.py` – Entry point script to run the full pipeline.
- `collect_data.py` – Scripts for downloading and preparing MIDI datasets.
- `preprocess.py` – Converts raw MIDI files into model‑ready representations.
- `train.py` – Trains the neural network model.
- `generate.py` – Generates new MIDI files from a trained model.
- `config.py` – Configuration file with hyper‑parameters.
- `requirements.txt` – Python dependencies.
- `templates/` – HTML templates for the optional web UI.
- `static/` – CSS and JS assets for the web UI.

## Getting Started
1. **Clone the repo**
   ```bash
   git clone https://github.com/Harisaran07/codealpha-ai-music-generator.git
   cd codealpha-ai-music-generator
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the pipeline**
   ```bash
   python app.py --all
   ```
   This will download data, preprocess, train, and generate music.

## Usage
- To **download data only**:
  ```bash
  python collect_data.py
  ```
- To **preprocess** after data download:
  ```bash
  python preprocess.py
  ```
- To **train** the model:
  ```bash
  python train.py
  ```
- To **generate** new music after training:
  ```bash
  python generate.py
  ```

## License
MIT License – see the `LICENSE` file for details.

## Contributing
Feel free to open issues or submit pull requests. Contributions are welcome!
