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
pip install --upgrade pip
pip install mediapipe opencv-python matplotlib scikit-learn seaborn joblib Pillow pandas numpy
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

#### Data Processing:
```bash
# Extract pose landmarks from training images
python src/load_yoga_training_data.py

# Convert landmark data to CSV format
python src/make_pose_dataset_csv.py

# Create angle-based feature dataset
python src/make_angles_dataset_csv.py
```

#### Training and Testing:
```bash
# Train the pose classification model
python src/train_model.py

# Test model with live webcam input
python src/test_model.py
```

#### Utilities:
```bash
# Add new training images
python src/add_new_files.py

# Detect pose in single image
python src/detect_image.py
```

Press **Q** to quit webcam mode.

---

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

