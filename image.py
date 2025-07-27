import mediapipe as mp 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as pimg

# Create the pose detection model
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.9,
min_tracking_confidence=0.7)
# where the min_detection_conficence and min_tracking_confidence are the minimum threshold values for detecting the pose


# Read image
img = cv2.imread("training-data/downward_dog_1.png")  # default: BGR with 3 channels

# Convert from BGR to RGB as mediapipe (mp) expects RGB input. 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Pass to MediaPipe
results = mp_pose.process(img_rgb)


print(results.pose_landmarks)