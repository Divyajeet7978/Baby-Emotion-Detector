---

# Baby Emotion Recognition System

## 🔍 Overview
The **Facial Emotion Recognition System** is an advanced image classification model powered by deep learning techniques. This system is designed to detect and classify a baby’s emotional state based on facial expressions. By utilizing **Convolutional Neural Networks (CNNs)**, the model processes grayscale images to predict emotions, including **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**.

This project serves as a valuable tool for parenting assistance, medical monitoring, and developing interactive baby care solutions.

---

## 📂 Project Structure
```plaintext
baby-emotion-recognition/
├── templates/         # HTML templates for the web interface
├── uploads/           # Directory for uploaded images
├── .gitattributes     # Configuration for text file attributes
├── .gitignore         # Specifies ignored files for version control
├── app.py             # Core Flask application for emotion recognition
├── BabyEmotion.h5     # Trained model stored in H5 format
├── BabyEmotion.json   # JSON representation of the model architecture
├── Detection.py       # Script for facial emotion detection
├── detection_app.py   # Real-time emotion recognition via webcam
├── LossGraph.py       # Script for visualizing training loss and accuracy
├── predict.py         # Single-image emotion prediction script
├── README.md          # Comprehensive project documentation
├── requirements.txt   # List of dependencies
├── setup.bat          # Batch file to initialize the project environment
├── training_model.py  # Model training script
```

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd baby-emotion-recognition
```

### 2️⃣ Create a Virtual Environment
For Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 🎥 Run the Application
Start the Flask web application:
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

To perform real-time emotion analysis:
- Click the **Start Detection** button to begin analyzing baby emotions in real-time.

### 🎭 Making Predictions
#### Single Image Emotion Detection:
Run the `predict.py` script with the desired image path:
```bash
python predict.py
```

#### Real-Time Webcam Detection:
Run the `detection_app.py` script:
```bash
python detection_app.py
```

This activates your webcam and performs real-time emotion recognition.

---

## 💡 Model Details
The deep learning model is built using **Convolutional Neural Networks (CNNs)**. Here are the key specifications:

- **Input**: Grayscale images of baby faces (48×48 pixels resolution).
- **Output**: Classification of emotions into one of seven categories:
  - 😠 **Angry**
  - 🤢 **Disgust**
  - 😨 **Fear**
  - 😃 **Happy**
  - 😐 **Neutral**
  - 😢 **Sad**
  - 😲 **Surprise**

---

## 📊 Training the Model
To train the model:
1. Open the `training_model.py` file to adjust training parameters if required.
2. Run the script:
```bash
python training_model.py
```
The trained model will be saved as `BabyEmotion.h5`.

To visualize training progress, run:
```bash
python LossGraph.py
```

---

## 🔧 Dependencies
This project requires the following libraries:
- **Flask**: Web application framework.
- **TensorFlow/Keras**: Deep learning toolkit.
- **OpenCV**: Image processing library.
- **Matplotlib**: Visualization of loss and accuracy.
- Additional dependencies can be found in `requirements.txt`.

Install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributions
We welcome contributions to improve and expand the project. To contribute:
1. Fork the repository.
2. Make your changes or add new features.
3. Submit a pull request with detailed explanations.

---
