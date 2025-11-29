import sys
import cv2
import joblib
import mediapipe as mp
import numpy as np
from collections import deque
from .utils import compute_feature_vector_from_points
import math

mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]


def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Return the angle ABC in degrees given 3 points (2D).
    a, b, c are [x, y] vectors.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


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

            sys.modules["__main__"].featurize_dataframe = featurize_dataframe  # type: ignore[attr-defined]

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
            cue = ""  # reset cue each frame

            if res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

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

                # ------------- Pose-specific feedback -------------
                # Common body points
                shoulders = (
                    pts[mp_pose.PoseLandmark.LEFT_SHOULDER.value, :2]
                    + pts[mp_pose.PoseLandmark.RIGHT_SHOULDER.value, :2]
                ) / 2.0
                hips = (
                    pts[mp_pose.PoseLandmark.LEFT_HIP.value, :2]
                    + pts[mp_pose.PoseLandmark.RIGHT_HIP.value, :2]
                ) / 2.0
                knees = (
                    pts[mp_pose.PoseLandmark.LEFT_KNEE.value, :2]
                    + pts[mp_pose.PoseLandmark.RIGHT_KNEE.value, :2]
                ) / 2.0
                ankles = (
                    pts[mp_pose.PoseLandmark.LEFT_ANKLE.value, :2]
                    + pts[mp_pose.PoseLandmark.RIGHT_ANKLE.value, :2]
                ) / 2.0

                # ---- DOWNWARD DOG ----
                if label_to_show == "downward_dog":
                    hip_angle = angle_3pts(shoulders, hips, ankles)
                    if hip_angle < 80:
                        cue = "Lift your hips up"
                    elif hip_angle > 120:
                        cue = "Walk your feet back / straighten legs"
                    else:
                        cue = "Nice inverted V shape!"

                # ---- PLANK ----
                elif label_to_show == "plank":
                    hip_angle = angle_3pts(shoulders, hips, ankles)
                    if hip_angle < 160:
                        cue = "Lift your hips in line with shoulders"
                    else:
                        cue = "Strong straight plank!"

                # ---- CHILD'S POSE ----
                elif label_to_show == "childs_pose":
                    fold_angle = angle_3pts(shoulders, hips, knees)
                    if fold_angle > 110:
                        cue = "Sink hips back towards your heels"
                    else:
                        cue = "Relax into the fold and breathe"

                # ---- HALF-BOAT ----
                elif label_to_show == "half_boat":
                    hip_angle = angle_3pts(shoulders, hips, ankles)
                    if hip_angle > 110:
                        cue = "Lift your legs a bit higher"
                    elif hip_angle < 50:
                        cue = "Lean your torso slightly back"
                    else:
                        cue = "Great strong V-shape!"

            # ---------- Draw label + cue ----------
            cv2.putText(
                frame,
                f"{label_to_show} ({conf * 100:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                2,
            )

            if cue:
                cv2.putText(
                    frame,
                    cue,
                    (10, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Yoga Pose Detection (Press Q to Quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
