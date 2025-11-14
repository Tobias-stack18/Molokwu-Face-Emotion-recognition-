
import os
import io
import base64
import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, jsonify, g
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

DB_PATH = 'database.db'
MODEL_PATH = 'model.h5'
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB

# Load model
if not os.path.exists(MODEL_PATH):
    print("Warning: model.h5 not found. Please run model.py to create it.")
    model = None
else:
    model = load_model(MODEL_PATH)
    print("Loaded model.h5")

# face detector (Haar cascade)
cascPath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascPath)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
    return db

def init_db():
    db = get_db()
    cur = db.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS uses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT,
            image BLOB,
            result TEXT
        )
    ''')
    db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_face(face_img):
    # face_img: grayscale numpy array of a face (48x48 ideally)
    face = cv2.resize(face_img, (48,48))
    face = face.astype("float")/255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

def predict_emotion_from_image_array(img_array):
    # img_array: BGR color image (opencv)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48,48))
    if len(faces) == 0:
        return {"error": "no_face_detected"}
    # take largest face
    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
    face = gray[y:y+h, x:x+w]
    processed = preprocess_face(face)
    preds = model.predict(processed)[0]
    top_idx = np.argmax(preds)
    return {
        "emotion": EMOTIONS[top_idx],
        "scores": {EMOTIONS[i]: float(preds[i]) for i in range(len(preds))},
        "bbox": [int(x), int(y), int(w), int(h)]
    }

@app.route('/')
def index():
    init_db()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name', 'Anonymous')
    # either file upload or base64 image (from webcam)
    file = request.files.get('image')
    img = None
    if file and allowed_file(file.filename):
        img_bytes = file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    else:
        # check for base64 webcam
        data_url = request.form.get('camera_image')
        if data_url:
            header, encoded = data_url.split(',',1)
            img_bytes = base64.b64decode(encoded)
            npimg = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "no_image_received"}), 400
    if model is None:
        return jsonify({"error": "model_not_loaded"}), 500

    result = predict_emotion_from_image_array(img)
    # Save to DB: store the original image as JPEG blob + result JSON
    _, img_encoded = cv2.imencode('.jpg', img)
    img_blob = sqlite3.Binary(img_encoded.tobytes())

    db = get_db()
    cur = db.cursor()
    cur.execute('INSERT INTO uses (name, timestamp, image, result) VALUES (?, ?, ?, ?)',
                (name, datetime.utcnow().isoformat(), img_blob, str(result)))
    db.commit()

    return jsonify(result)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
