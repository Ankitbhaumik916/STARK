import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Gesture mapping
GESTURES = {
    "00000": "Fist ✊",
    "01000": "Thumbs Up 👍",
    "01110": "Peace ✌",
    "11111": "Open Palm ✋",
    "10001": "Rock On 🤘",
}

def get_finger_state(landmarks):
    """
    Returns a binary string representing which fingers are open.
    Order: Thumb, Index, Middle, Ring, Pinky (1=open, 0=closed)
    """
    fingers = ""

    # Detect hand orientation (left or right)
    wrist_x = landmarks[0][0]  # Wrist X position
    thumb_tip_x = landmarks[4][0]
    thumb_base_x = landmarks[3][0]

    is_right_hand = thumb_tip_x > wrist_x  # Right hand if thumb is on the right
    is_left_hand = not is_right_hand

    # **Thumb Detection Fix**
    if is_right_hand:
        fingers += "1" if thumb_tip_x > thumb_base_x else "0"
    else:
        fingers += "1" if thumb_tip_x < thumb_base_x else "0"

    # Other fingers (Tip Y < Base Y means OPEN)
    for tip, base in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers += "1" if landmarks[tip][1] < landmarks[base][1] else "0"

    return fingers

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

while cap.isOpened():
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

            # Get finger states
            finger_state = get_finger_state(landmarks)

            # Recognize gesture
            gesture_name = GESTURES.get(finger_state, "Unknown Gesture")

            # Display gesture
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Gesture Recognition", frame)

    # Quit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
