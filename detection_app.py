import os
import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

MODEL_PATH = "BabyEmotion.h5"
model = load_model(MODEL_PATH)
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

video = cv2.VideoCapture(0)

def gen_frames():
    """Video streaming generator function.
       Captures frames from the webcam, detects faces, predicts emotion,
       and yields the processed frames as JPEG bytes.
    """
    while True:
        success, frame = video.read()  # Read a frame from the webcam
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                except Exception as e:
                    continue

                img_pixels = roi_gray.astype('float32') / 255.0
                img_pixels = np.reshape(img_pixels, (1, 48, 48, 1))
                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions[0])
                predicted_emotion = CLASS_LABELS[max_index]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, predicted_emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Home page with a button to start the detection."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. The browser will hit this endpoint to retrieve the stream."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the app on localhost:5000
    app.run(debug=True)
