from flask import Flask, render_template, request
import os
from detect import detect_motion

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    detect_motion(0)
    return "Webcam detection stopped."

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    detect_motion(filepath)
    return "Video processing done."

if __name__ == '__main__':
    app.run(debug=True)
