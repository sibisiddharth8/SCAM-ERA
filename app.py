from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from datetime import datetime
import json
from time import time as current_time
import importlib

app = Flask(__name__)

# Directories for storing videos
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Allowed file extensions for video uploads
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

def allowed_file(filename):
    """Check if the uploaded file is a valid video file."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        timestamp = int(current_time())
        uploaded_filename = f"uploaded_video_{timestamp}.mp4"
        uploaded_video_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
        file.save(uploaded_video_path)

        processed_filename = f"processed_video_{timestamp}.mp4"
        processed_video_path = os.path.join(app.config['OUTPUT_FOLDER'], processed_filename)

        try:
            module = importlib.import_module("deepfake_detector")
            function = getattr(module, "run")
            result_from_det = function(uploaded_video_path, processed_video_path)
        except Exception as e:
            app.logger.error(f"Error processing video: {e}")
            return jsonify({'error': 'Processing failed. Check the server logs.'}), 500

        video_info = {
            'name': file.filename,
            'size': f"{os.path.getsize(uploaded_video_path) / 1024:.2f} KB",
            'uploaded_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'authenticity_score': result_from_det,
            'uploaded_video_path': f'/static/uploads/{uploaded_filename}',
            'processed_video_path': f'/static/outputs/{processed_filename}'
        }

        return jsonify({'video_info': video_info})

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/result')
def result():
    video_info = request.args.get('video_info', {})
    return render_template('result.html', video_info=json.loads(video_info))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

