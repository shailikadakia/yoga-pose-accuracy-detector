import mediapipe as mp 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import json
import os 
from glob import glob
from utils import normalize_landmarks
from detect_image import detect_pose

# Establish paths
input_root = "../training-data"
output_root = "../data"
os.makedirs(output_root, exist_ok=True)


# List all the folders in training-data folder 
for pose_folder in os.listdir(input_root):
  label = pose_folder 
  folder_path = os.path.join(input_root, pose_folder) # Create a path like training-data/plank

  # Skip files that aren't folders 
  if not os.path.isdir(folder_path):
    continue

  # Glob is used to get a list of all png image paths inside this path 
  for image_path in glob(os.path.join(folder_path, "*.png")):
    img = cv2.imread(image_path)  
    landmarks = detect_pose(img, False)
    data = {
      "label": label,
      "landmarks": landmarks
    }
      
      # os.path.splittext removes the file extensions 
      # os.path.basename gets the fiename only so plank_1.png
    base_name = os.path.splitext(os.path.basename(image_path))[0]

      # Create output folder
    output_path = os.path.join(output_root, f"{base_name}.json")
    with open(output_path, "w") as f:
      json.dump(data, f, indent=2)

      print(f"✅ Saved {output_path}")
  else:
    print(f"⚠️ No pose detected in {image_path}")