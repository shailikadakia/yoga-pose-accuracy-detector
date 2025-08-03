import mediapipe as mp 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import json
import os 
from glob import glob
from utils import normalize_landmarks

# Establish paths
input_root = "training-data"
output_root = "data"
os.makedirs(output_root, exist_ok=True)


# Create the pose detection model
mp_pose = mp.solutions.pose.Pose(static_image_mode = True, min_detection_confidence=0.9, min_tracking_confidence=0.7)
# where the min_detection_conficence and min_tracking_confidence are the minimum threshold values for detecting the pose

# List all the folders in training-data folder 
for pose_folder in os.listdir(input_root):
  label = pose_folder 
  folder_path = os.path.join(input_root, pose_folder) # Create a path like training-data/plank

  # Skip files that aren't folders 
  if not os.path.isdir(folder_path):
    continue

  # Glob is used to get a list of all png image paths inside this path 
  for image_path in glob(os.path.join(folder_path, "*.png")):
    img = cv2.imread(image_path)  # default: BGR with 3 channels
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(img_rgb)
    if results.pose_landmarks: 
      keypoints = normalize_landmarks(results.pose_landmarks.landmark)
      data = {
        "label": label,
        "landmarks": keypoints
      }
      
      # os.path.splittext reoves the file extensions 
      # os.path.basename gets the fiename only so plank_1.png
      base_name = os.path.splitext(os.path.basename(image_path))[0]

      # Create output folder
      output_path = os.path.join(output_root, f"{base_name}.json")
      with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

        print(f"✅ Saved {output_path}")
    else:
      print(f"⚠️ No pose detected in {image_path}")