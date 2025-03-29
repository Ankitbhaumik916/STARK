import torch
import cv2
import numpy as np

# Load MoveNet model (Make sure you have the model downloaded)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MoveNet model (Replace with the correct PyTorch model path)
model_path = "movenet.pth"
model = torch.load(model_path, map_location=device)
model.eval().to(device)  # Move model to GPU

# Function to preprocess image
def preprocess(image):
    image = cv2.resize(image, (192, 192))  # Resize
    image = image.astype(np.float32) / 255.0  # Normalize
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)  # Convert to tensor
    return image

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    input_tensor = preprocess(frame_rgb)  # Preprocess image

    with torch.no_grad():  # Disable gradient calculations for speed
        keypoints = model(input_tensor)

    keypoints = keypoints.cpu().numpy()  # Move data back to CPU for OpenCV
    print(keypoints)  # Debugging output

    cv2.imshow("MoveNet Pose Detection (GPU)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
