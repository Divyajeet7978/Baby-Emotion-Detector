import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import logging
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Model configuration
MODEL_PATH = "BabyEmotion.h5"
model = load_model(MODEL_PATH)
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Camera management
video = None
camera_active = False

def get_camera():
    global video
    if video is None or not video.isOpened():
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("❌ Unable to access primary camera! Trying secondary index.")
            video = cv2.VideoCapture(1)
    return video

def release_camera():
    global video, camera_active
    if video is not None and video.isOpened():
        video.release()
        video = None
        camera_active = False
        print("✅ Camera released successfully.")

def gen_frames():
    global camera_active
    camera = get_camera()
    camera_active = True
    
    while camera_active:
        success, frame = camera.read()
        if not success:
            print("❌ Camera read failed! Attempting to reinitialize...")
            release_camera()
            camera = get_camera()
            continue

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = roi_gray.astype('float32') / 255.0
                img_pixels = np.reshape(img_pixels, (1, 48, 48, 1))
                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions[0])
                predicted_emotion = CLASS_LABELS[max_index]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, predicted_emotion, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Frame Processing Error: {e}")
            continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    release_camera()
    return jsonify({"message": "Camera stopped successfully"})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    release_camera()

    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return jsonify({'error': 'Unsupported file format'}), 400

        image = cv2.resize(image, (48, 48))
        img_pixels = image.astype('float32') / 255.0
        img_pixels = np.reshape(img_pixels, (1, 48, 48, 1))
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = CLASS_LABELS[max_index]

        return jsonify({
            "emotion": predicted_emotion,
            "recommendation": f"The detected emotion is {predicted_emotion}. Please take appropriate action!"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camera_status', methods=['GET'])
def camera_status():
    global video, camera_active
    status = {
        'active': camera_active,
        'camera_available': video.isOpened() if video else False
    }
    return jsonify(status)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)