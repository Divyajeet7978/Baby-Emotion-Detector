import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the model and class labels
MODEL_PATH = "BabyEmotion.h5"
model = load_model(MODEL_PATH)
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Initialize webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("❌ Unable to access the camera! Trying another index.")
    video = cv2.VideoCapture(1)

# Ensure the camera is released when stopping the feed
def release_camera():
    global video
    if video.isOpened():
        video.release()
        print("✅ Camera released successfully.")

# Generate frames for live feed
def gen_frames():
    global video
    while True:
        success, frame = video.read()
        if not success:
            print("❌ Camera read failed!")
            break
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                except Exception as e:
                    print(f"Resize Error: {e}")
                    continue

                img_pixels = roi_gray.astype('float32') / 255.0
                img_pixels = np.reshape(img_pixels, (1, 48, 48, 1))
                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions[0])
                predicted_emotion = CLASS_LABELS[max_index]

                # Draw face rectangle and emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, predicted_emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Encode frame and return it
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Frame encoding failed!")
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Frame Processing Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stops the live feed camera."""
    release_camera()
    return jsonify({"message": "Camera stopped successfully"})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Processes uploaded image for emotion detection."""
    release_camera()  # Ensure the camera is released before processing an image

    if 'image' not in request.files:
        print("❌ No image uploaded! Check request format.")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    print(f"📸 Received file: {file.filename}")

    if file.filename == '':
        print("❌ Empty filename received.")
        return jsonify({'error': 'Empty filename'}), 400

    try:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        print(f"✅ Image saved at {file_path}")

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("❌ Unsupported image format!")
            return jsonify({'error': 'Unsupported file format'}), 400

        image = cv2.resize(image, (48, 48))
        img_pixels = image.astype('float32') / 255.0
        img_pixels = np.reshape(img_pixels, (1, 48, 48, 1))
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = CLASS_LABELS[max_index]

        print(f"🎭 Predicted Emotion: {predicted_emotion}")
        return jsonify({
            "emotion": predicted_emotion,
            "recommendation": f"The detected emotion is {predicted_emotion}. Please take appropriate action!"
        })
    except Exception as e:
        print(f"❌ Error during image processing: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)