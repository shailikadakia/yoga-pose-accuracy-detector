import sys
import cv2
import joblib
import mediapipe as mp
import numpy as np
from collections import deque
from .utils import compute_feature_vector_from_points  

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def run_webcam(conf_threshold: float = 0.50, history_len: int = 8) -> None:
    """
    Open webcam, run pose detection + KNN classification loop.
    Press 'q' to quit.
    """
    # -------- Load model --------
    try:
        rt = joblib.load("./src/pose_knn_runtime.pkl")
        print("Loaded pose_knn_runtime.pkl")
        scaler, knn, label_encoder = rt["scaler"], rt["knn"], rt["label_encoder"]
    except Exception as e_runtime:
        print("Runtime bundle not found or failed to load:", e_runtime)
        try:
            def featurize_dataframe(X_df, *args, **kwargs):
                return X_df
            sys.modules['__main__'].featurize_dataframe = featurize_dataframe

            bundle = joblib.load("./src/pose_knn_bundle.pkl")
            print("Loaded legacy pose_knn_bundle.pkl")
            pipe = bundle["pipe"]
            scaler = pipe.named_steps["scaler"]
            knn = pipe.named_steps["knn"]
            label_encoder = bundle["label_encoder"]
        except Exception as e_legacy:
            raise SystemExit(f"Failed to load any bundle: {e_legacy}")

    # -------- Open camera --------
    cap = None
    for idx in (0, 1, 2, 3):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print(f"Using camera index {idx}")
            break
    if cap is None or not cap.isOpened():
        raise SystemExit("Camera failed to open on indices 0–3.")

    pred_hist = deque(maxlen=history_len)

    # -------- Pose loop --------
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Can't receive frame.")
                break

            # run mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)
            rgb.flags.writeable = True

            label_to_show, conf = "Unknown", 0.0

            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                pts = np.array(
                    [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark],
                    dtype=float,
                )
                feats = compute_feature_vector_from_points(pts).reshape(1, -1)

                Xs = scaler.transform(feats)
                proba = knn.predict_proba(Xs)[0]
                idx = int(np.argmax(proba))
                conf = float(proba[idx])
                raw_label = label_encoder.classes_[idx]

                pred_hist.append(raw_label)
                smoothed = max(set(pred_hist), key=pred_hist.count)
                label_to_show = smoothed if conf >= conf_threshold else "Unknown"

            cv2.putText(frame, f"{label_to_show} ({conf:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Yoga Pose Detection (Press Q to Quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

