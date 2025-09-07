import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load pipeline + encoder from bundle
bundle = joblib.load("pose_knn_bundle.pkl")
pipe = bundle["pipe"]                # Pipeline (scaler + KNN)
le   = bundle["label_encoder"]       # LabelEncoder

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera failed to open.")
    exit()

with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame.")
            break

        # Convert to RGB for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True

        # Convert back for display
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Flatten pose landmarks (x,y,z → 99 features)
            landmarks = results.pose_landmarks.landmark
            feats = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten().reshape(1, -1)

            # Predict using pipeline (scaler inside pipe)
            if feats.shape[1] == pipe.named_steps["knn"].n_features_in_:
                y_idx = pipe.predict(feats)[0]
                label = le.inverse_transform([y_idx])[0]

                cv2.putText(frame, f'Pose: {label}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Yoga Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
