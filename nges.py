import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

# Capture video feed
cap = cv2.VideoCapture(0)

def get_wrist_rotation(hand_landmarks):
    """Calculate wrist rotation angle using index and pinky finger alignment."""
    index_x = hand_landmarks.landmark[8].x
    pinky_x = hand_landmarks.landmark[20].x
    wrist_x = hand_landmarks.landmark[0].x  # Wrist position

    # Compare index & pinky positions relative to wrist to detect rotation
    if index_x > wrist_x and pinky_x < wrist_x:
        return "clockwise"  # Rotate right → Volume Up
    elif index_x < wrist_x and pinky_x > wrist_x:
        return "counterclockwise"  # Rotate left → Volume Down
    return None

def adjust_volume(rotation):
    """Adjust system volume based on hand rotation."""
    if rotation == "clockwise":
        pyautogui.press("volumeup")
    elif rotation == "counterclockwise":
        pyautogui.press("volumedown")

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            rotation = get_wrist_rotation(hand_landmarks)
            adjust_volume(rotation)  # Adjust volume based on rotation

    cv2.imshow("Twist Volume Control", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
