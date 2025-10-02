import cv2
import numpy as np
from PIL import Image

video_path = r"C:\Users\toazb\Documents\GitHub\race_simulation\videos\edge_cases\collision_scalar_success.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Could not open video")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
positions = [0, 210, frame_count - 1]
frames = []

for pos in positions:
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    success, frame = cap.read()
    if success:
        frames.append(frame)

cap.release()

for i, frame in enumerate(frames):
    # Denoise to reduce compression artifacts
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img.save(f"frames\\frame{i}.png", quality=100, subsampling=0)

print("Frames saved")
