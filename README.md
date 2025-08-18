# Yoga Pose Accuracy Detector

This project uses **MediaPipe** and **OpenCV** to detect human pose landmarks from images or webcam input. It is designed to help identify and visualize yoga poses in real time.

---

## 🔧 Setup Instructions (Virtual Environment)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/yoga-pose-accuracy-detector.git
cd yoga-pose-accuracy-detector
```


### 2. Create and Activate a Virtual Environment
#### MacOS: 
```bash
python3.10 -m venv mp-env
source mp-env/bin/activate
```

#### Windows
```bash
python -m venv mp-env
mp-env\Scripts\activate
```

#### Install Dependencies 
```bash
pip install --upgrade pip
pip install mediapipe opencv-python matplotlib scikit-learn
```

### 3. Running Scripts 
```bash
python image.py
python webcam.py
python load_yoga_training_data.py
```
Press Q to quit