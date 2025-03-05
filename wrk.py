import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading

# Initialize MediaPipe Hands with optimizations
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7, 
    static_image_mode=False,  # Keeps tracking hands between frames
    max_num_hands=1  # Optimize for single-hand use
)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Open Webcam with lower resolution for faster processing
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Function for capturing frames in a separate thread
frame_lock = threading.Lock()
latest_frame = None

def capture_frames():
    global latest_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            latest_frame = frame

# Start frame capture thread
threading.Thread(target=capture_frames, daemon=True).start()

prev_time = 0  # For FPS calculation

while True:
    if latest_frame is None:
        continue

    with frame_lock:
        frame = latest_frame.copy()

    # Flip and convert color format
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(rgb_frame)

    # Draw hand landmarks and control mouse
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract fingertip coordinates
            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            middle_finger_tip = hand_landmarks.landmark[12]

            x, y = int(index_finger_tip.x * screen_w), int(index_finger_tip.y * screen_h)
            pyautogui.moveTo(x, y, duration=0.05)  # Faster movement

            # Compute distances for clicks
            thumb_index_dist = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y])
            )
            thumb_middle_dist = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_finger_tip.x, middle_finger_tip.y])
            )

            # Left Click: Pinch Index & Thumb
            if thumb_index_dist < 0.03:
                pyautogui.click()

            # Right Click: Pinch Middle & Thumb
            if thumb_middle_dist < 0.03:
                pyautogui.rightClick()

            # Scroll: Move Hand Up/Down
            if index_finger_tip.y < 0.4:  # Hand moved up
                pyautogui.scroll(10)
            elif index_finger_tip.y > 0.6:  # Hand moved down
                pyautogui.scroll(-10)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Fast Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
