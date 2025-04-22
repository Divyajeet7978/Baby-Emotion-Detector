Baby Emotion Recognition System

🔍 Overview
The Baby Emotion Recognition System is a deep learning-powered image classification model designed to detect and classify a baby’s emotional state based on facial expressions. This system leverages Convolutional Neural Networks (CNNs) to analyze grayscale images and predict emotions such as Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
This project aims to provide a reliable and efficient emotion recognition system applicable to parenting assistance, medical monitoring, and interactive baby care solutions.

📂 Project Structure

baby-emotion-recognition/
├── templates/         # HTML templates for the Flask web app
├── uploads/           # Directory for uploaded images
├── .gitattributes     # Git attributes file
├── .gitignore         # Ignore unnecessary files in version control
├── app.py             # Main Flask application for emotion detection
├── BabyEmotion.h5     # Trained model saved in H5 format
├── BabyEmotion.json   # Model architecture in JSON format
├── Detection.py       # Python script for emotion detection from images
├── detection_app.py   # Video-based emotion recognition using webcam
├── LossGraph.py       # Visualization of training loss over epochs
├── predict.py         # Python script for emotion prediction on a single image
├── Readme.md          # Project documentation
├── requirements.txt   # Dependencies needed to run the project
├── setup.bat          # Script for setting up the environment
├── training_model.py  # Python script for training the emotion detection model


🚀 Installation & Setup
1️⃣ Clone the Repository
git clone <repository-url>
cd baby-emotion-recognition

2️⃣ Create a Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate

Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

🎥 Run the Application
Start the Flask web app:
python detection_app.py

Once the server starts, open the browser and go to:
http://127.0.0.1:5000

Click "Start Detection" to analyze baby emotions in real-time.

🎭 Making Predictions
For Single Image Emotion Detection
Run the predict.py script with an image path:
python predict.py

For Real-Time Webcam Detection
Run detection_app.py:
python detection_app.py

This will activate the webcam and display real-time emotion predictions.

💡 Model Details
- Architecture: CNN (Convolutional Neural Network)
- Input: Grayscale baby face images (48×48 pixels)
- Output: Emotion classification (7 categories)
- Supported Emotions:- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😃 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

📊 Training the Model
- Open training_model.py to modify training parameters if needed.
- Run the script:python training_model.py

- The trained model will be saved as BabyEmotion.h5.
- Use LossGraph.py to visualize training performance.

🔧 Dependencies
Ensure the following libraries are installed:
- Flask – Web application framework
- TensorFlow/Keras – Deep learning framework
- OpenCV – Image processing
- Matplotlib – Plotting loss graphs
- Other dependencies listed in requirements.txt

🤝 Contributing
We welcome contributions! To contribute:
- Fork the repository.
- Make improvements or fix bugs.
- Submit a pull request with a clear description.

📜 License
This project is open-source under the MIT License. Feel free to modify and use it for research, educational, or commercial applications.
