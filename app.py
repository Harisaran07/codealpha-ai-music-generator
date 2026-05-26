import os
from flask import Flask, render_template, jsonify, send_file
import config
from generate import generate_music

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def run_generation():
    try:
        # Run the generation script
        # Note: In a real app, this should be a background task because it takes a while,
        # but for this demo, we'll block the request and return the result.
        output_file = generate_music(temperature=1.0, length=200)
        
        if output_file and os.path.exists(output_file):
            filename = os.path.basename(output_file)
            return jsonify({"status": "success", "file": filename})
        else:
            return jsonify({"status": "error", "message": "Generation failed."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/midi/<filename>')
def serve_midi(filename):
    midi_path = os.path.join(config.OUTPUT_DIR, filename)
    if os.path.exists(midi_path):
        return send_file(midi_path, mimetype="audio/midi")
    return "File not found", 404

if __name__ == '__main__':
    # Make sure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    app.run(debug=True, port=5000)
