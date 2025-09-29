# Yoga Pose Accuracy Detector

This project uses **MediaPipe** and **OpenCV** to detect human pose landmarks from images or webcam input. It is designed to help identify and visualize yoga poses in real time using machine learning classification.

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

### 3. Install Dependencies 
```bash
make install
```

#### Required Libraries:
- **mediapipe** - Pose detection and landmark extraction
- **opencv-python** - Computer vision and image processing
- **matplotlib** - Data visualization and plotting
- **scikit-learn** - Machine learning algorithms (KNN classifier, preprocessing)
- **seaborn** - Statistical data visualization
- **joblib** - Model serialization and loading
- **Pillow** - Image processing and format conversion
- **pandas** - Data manipulation and CSV handling
- **numpy** - Numerical computing and array operations

### 4. Running Scripts 
```bash
make prepare   # Load training data and build CSV datasets
make train     # Train the KNN model (runs prepare first if needed)
make test      # Run live webcam detection (press 'q' to quit)
make clean     # Remove generated files and caches
```

## 📁 Project Structure

- **src/** - Source code files
- **data/** - JSON files with pose landmark data
- **training-data/** - Training images organized by pose type
- **models/** - Trained model files
- **pose_dataset.csv** - Raw landmark coordinates
- **pose_angles_dataset.csv** - Processed angle features

---

## 🧘‍♀️ Supported Poses

The model can classify various yoga poses based on the training data in your `training-data/` folder.

---

## 📊 Data Sources & Credits

This project uses publicly available datasets and reference images for training and evaluation:

- [Yoga Posture Dataset (Kaggle, Mrinal Tyagi)](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset/data?select=Adho+Mukha+Svanasana)  
- [Yoga Poses Dataset (Kaggle, by Niharika Pandit)](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset/data)  
- [Yoga With Adriene YouTube Channel](https://www.youtube.com/user/yogawithadriene) — select reference images used to supplement training data.

All credit goes to the original dataset creators and Yoga With Adriene.  
This project is for **educational and research purposes only** and not intended for commercial use.

