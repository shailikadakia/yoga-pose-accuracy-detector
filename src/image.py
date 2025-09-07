import mediapipe as mp 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as pimg
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

# Create the pose detection model
mp_pose = mp.solutions.pose.Pose(
  min_detection_confidence=0.9,  
  min_tracking_confidence=0.7,
  )
# where the min_detection_conficence and min_tracking_confidence are the minimum threshold values for detecting the pose

image_path = "../training-data/cat/cat_1.png"

def detect_image(image_path):
    # Read image
  img = cv2.imread(image_path)  # default: BGR with 3 channels

  # Convert from BGR to RGB as mediapipe (mp) expects RGB input. 
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # Pass to MediaPipe
  results = mp_pose.process(img_rgb)
  # print(results.pose_landmarks)

  annotated_image = img_rgb.copy()

  # Loop through the detected poses to visualize.
  if results.pose_landmarks: 
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      results.pose_landmarks,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image, results


def visualize_landmarks(org_image, annotated_image):
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Pose Estimation')
    plt.imshow(annotated_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def extract_keypoints(landmarks):
  if landmarks:
    keypoints = {}
    for idx, landmark in enumerate(landmarks.pose_landmarks.landmark):
      keypoints[mp_pose.PoseLandmark(idx).name] = {
        'x': landmark.x,
        'y': landmark.y,
        'z': landmark.z,
        'visibility': landmark.visibility
    }
    return keypoints
  return None

org_image = cv2.imread(image_path) 
annotated_image, results = detect_image(image_path)
visualize_landmarks(org_image, annotated_image)
keypoints = extract_keypoints(results)

if keypoints:
  print('Detected keypoints')
  for name, details in keypoints:
    print(f"{name}: {details}")
