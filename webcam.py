import cv2
import mediapipe as mp

# Initialize MediaPipe pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Load the pose estimation model
# Min_dection_confidence: only shows results if detection is at least 90% confident 
# Min_tracking_confidence: tracks pose landmarks across frames if they're at least 70% confident. 
with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.7) as pose: 
    while cap.isOpened():
        # Ret is a boolean indicating if a frame was successfully read
        # Frame is actual image from webcam in BGR format 
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Convert the BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # Improves performance, tells mediapipe that image won't be modified, speeding up performance

        # Process the frame to detect pose
        results = pose.process(frame_rgb)

        # Draw the pose annotation on the original frame
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw green circles at each joint
        # Draw red lines connecting them 
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

        # Display the frame
        cv2.imshow('Pose Detection (Press Q to Quit)', frame_bgr)

        # Break on pressing 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
