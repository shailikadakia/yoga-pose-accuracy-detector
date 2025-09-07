import cv2
import mediapipe as mp
import numpy as np
import joblib
from detect_image import detect_pose

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera failed to open.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame.")
        break

    # Convert to RGB for mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = detect_pose(image, display=False, return_results=True)
    image.flags.writeable = True

    # Convert back for display
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        
    cv2.imshow("Yoga Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
