'''
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
'''

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

from detect_image import detect_pose             # returns MediaPipe results
from utils import compute_feature_vector_from_points         # returns (feats, feat_names)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---- load trained bundle ----
bundle = joblib.load("pose_knn_bundle.pkl")
full_pipe = bundle["pipe"]                       # Pipeline( angles -> scaler -> knn )
label_encoder = bundle["label_encoder"]

# We'll SKIP the "angles" step since we're computing angles ourselves here:
scaler = full_pipe.named_steps["scaler"]
knn    = full_pipe.named_steps["knn"]

# Optional: smooth predictions to avoid flicker
pred_hist = deque(maxlen=8)

# Pick the right camera index on macOS if Continuity Camera grabs index 0
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise SystemExit("Camera failed to open.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Can't receive frame.")
        break

    # Get MediaPipe results (detect_pose converts to RGB internally)
    results = detect_pose(frame, display=False, return_results=True)

    if results and results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 33x3 array of (x,y,z)
        pts = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=float)

        # Compute the SAME angle/ratio feature vector you used in training
        feats, _ = compute_feature_vector_from_points(pts)          # shape: (14,)
        X = feats.reshape(1, -1)

        # Run through scaler + KNN (skip the 'angles' transformer here)
        Xs = scaler.transform(X)
        proba = knn.predict_proba(Xs)[0]
        pred_idx = int(np.argmax(proba))
        raw_label = label_encoder.classes_[pred_idx]
        conf = float(proba[pred_idx])

        # Smooth the label over a short window
        pred_hist.append(raw_label)
        smoothed = max(set(pred_hist), key=pred_hist.count)

        # Optional confidence gate
        label_to_show = smoothed if conf >= 0.5 else "Unknown"

        cv2.putText(frame, f"{label_to_show} ({conf:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    cv2.imshow("Yoga Pose Detection (KNN angles)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
