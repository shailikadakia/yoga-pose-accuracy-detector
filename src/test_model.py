# webcam_angles.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

from utils import compute_feature_vector_from_points  # returns a 1D np.array of features

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def load_model():
    try:
        rt = joblib.load("pose_knn_runtime.pkl")
        print("Loaded pose_knn_runtime.pkl")
        return rt["scaler"], rt["knn"], rt["label_encoder"]
    except Exception as e_runtime:
        print("Runtime bundle not found or failed to load:", e_runtime)
        try:
            import sys
            def featurize_dataframe(X_df, *args, **kwargs):  
                return X_df
            sys.modules['__main__'].featurize_dataframe = featurize_dataframe

            bundle = joblib.load("pose_knn_bundle.pkl")
            print("Loaded legacy pose_knn_bundle.pkl (using scaler+knn from its pipeline)")
            pipe = bundle["pipe"]
            scaler = pipe.named_steps["scaler"]
            knn = pipe.named_steps["knn"]
            le = bundle["label_encoder"]
            return scaler, knn, le
        except Exception as e_legacy:
            raise SystemExit(f"Failed to load any bundle: {e_legacy}")

scaler, knn, label_encoder = load_model()

# Open Camera (not iPhone camera)
def open_camera():
    for idx in (0, 1, 2, 3):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print(f"Using camera index {idx}")
            return cap
    raise SystemExit("Camera failed to open on indices 0–3.")

cap = open_camera()

pred_hist = deque(maxlen=8)

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Can't receive frame.")
            break

        # Run pose
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = pose.process(rgb)
        rgb.flags.writeable = True

        if res.pose_landmarks:
            # draw skeleton
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 33x3 array (x,y,z) in normalized coords
            pts = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark], dtype=float)

            # SAME features as training
            feats = compute_feature_vector_from_points(pts).reshape(1, -1)

            # scale + knn
            Xs = scaler.transform(feats)
            proba = knn.predict_proba(Xs)[0]
            idx = int(np.argmax(proba))
            conf = float(proba[idx])
            raw_label = label_encoder.classes_[idx]

            # smoothing
            pred_hist.append(raw_label)
            smoothed = max(set(pred_hist), key=pred_hist.count)

            # confidence gate 
            label_to_show = smoothed if conf >= 0.50 else "Unknown"

            cv2.putText(frame, f"{label_to_show} ({conf:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Yoga Pose Detection (angles + KNN)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
