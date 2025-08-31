import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model components
best_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Start webcam
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

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True

        # Back to BGR for OpenCV display
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Extract keypoints and flatten
            landmarks = results.pose_landmarks.landmark
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().reshape(1, -1)

            # ❗ Ensure shape matches what your model expects (e.g. 99 features)
            if keypoints.shape[1] == scaler.n_features_in_:
                scaled = scaler.transform(keypoints)
                pred = best_model.predict(scaled)
                label = label_encoder.inverse_transform(pred)[0]

                # Show label
                cv2.putText(frame, f'Pose: {label}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show webcam window
        cv2.imshow("Yoga Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
