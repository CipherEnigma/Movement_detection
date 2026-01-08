from flask import Flask, request, send_from_directory, redirect, url_for
import os
from scripts.detect import detect_motion
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXT = {'mp4','avi','mov','mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/')
def index():
    # minimal inline HTML form so no templates are required
    return """
    <html><head><title>Movement Detection Upload</title></head>
    <body>
      <h3>Upload a video (mp4/avi/mov/mkv)</h3>
      <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="video/*"/>
        <input type="submit" value="Upload & Process"/>
      </form>
      <p>Or visit /webcam to run webcam demo (server must have camera).</p>
    </body></html>
    """

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if not allowed_file(file.filename):
        return "File type not allowed", 400

    filename = secure_filename(file.filename)
    uid = uuid.uuid4().hex[:8]
    in_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{filename}")
    file.save(in_path)

    out_name = f"annotated_{uid}.mp4"
    out_path = os.path.join(RESULT_FOLDER, out_name)

    # run headless detection (may take time depending on video length)
    summary = detect_motion(in_path, output_path=out_path)

    # provide a simple response with link to result
    return f"""
    <html><body>
      <h3>Processing complete</h3>
      <p>Summary: {summary}</p>
      <p><a href="{url_for('get_result', filename=out_name)}">Download / Play annotated video</a></p>
      <p><a href="/">Process another video</a></p>
    </body></html>
    """

@app.route('/results/<path:filename>')
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=False)

@app.route('/webcam')
def webcam():
    # run webcam for a short demo and store in results; runs headless so no GUI
    uid = uuid.uuid4().hex[:8]
    out_name = f"webcam_{uid}.mp4"
    out_path = os.path.join(RESULT_FOLDER, out_name)
    # webcam index 0, capture a few seconds (max_frames can be tuned)
    summary = detect_motion(0, output_path=out_path, max_frames=300)
    return f"Webcam processed: {summary}. Result: <a href='{url_for('get_result', filename=out_name)}'>{out_name}</a>"

if __name__ == '__main__':
    # for local demo use debug mode ON; for classroom disable debug and consider binding to 0.0.0.0
    app.run(host='127.0.0.1', port=5000, debug=True)
