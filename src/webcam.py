import cv2
import mediapipe as mp
import numpy as np
from detect_image import detect_pose
from classify_pose import classifyPose

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Try another index if this picks your iPhone camera
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise SystemExit("Camera failed to open.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Can't receive frame.")
        break

    # Ask detect_pose to give you the MediaPipe results object
    results = detect_pose(frame, display=False, return_results=True)

    if results and results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Classify: pass the landmark LIST (not the NormalizedLandmarkList)
        frame, label = classifyPose(results.pose_landmarks.landmark, frame, display=False)

    cv2.imshow("Yoga Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
