import cv2
import numpy as np
from PIL import Image

def overlay_frames(video_path, output_path, color_priority=None, scale_factor=1, target_size=None):
    """
    Overlay frames with HSV ranges for Blue/Green + Gray included per frame.
    Then detect and repair blue/green boxes by filling their bounding regions
    with solid pure blue or green.
    """
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

    color_ranges = {
        "blue":   [(40, 40, 15), (170, 255, 255)],
        "green":  [(20, 50, 25),  (80, 255, 255)],
        "yellow": [(20, 100, 100), (35, 255, 255)],
    }

    # Gray in BGR space (± tolerance)
    def gray_mask(frame, tol=30):
        lower = np.array([169 - tol, 169 - tol, 169 - tol], dtype=np.uint8)
        upper = np.array([169 + tol, 169 + tol, 169 + tol], dtype=np.uint8)
        return cv2.inRange(frame, lower, upper) > 0

    if color_priority is None:
        color_priority = ["green", "blue"]

    priority_map = {color: i for i, color in enumerate(color_priority)}

    # Start with blank canvas
    overlay = np.zeros_like(frames[0], dtype=np.uint8)
    priority_layer = np.full(frames[0].shape[:2], 9999, dtype=np.int32)

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Priority colors (Blue + Green)
        for color in ["blue", "green"]:
            if color not in priority_map:
                continue
            lower, upper = color_ranges[color]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper)) > 0
            update_mask = mask & (priority_map[color] < priority_layer)
            for c in range(3):
                overlay[:, :, c][update_mask] = frame[:, :, c][update_mask]
            priority_layer[update_mask] = priority_map[color]

        # Gray (BGR tolerance) — included for every frame
        if "gray" in priority_map:
            mask = gray_mask(frame)
            update_mask = mask & (priority_map["gray"] < priority_layer)
            for c in range(3):
                overlay[:, :, c][update_mask] = frame[:, :, c][update_mask]
            priority_layer[update_mask] = priority_map["gray"]

        # Default: fill remaining non-black pixels
        default_mask = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 0) & (priority_layer == 9999)
        for c in range(3):
            overlay[:, :, c][default_mask] = frame[:, :, c][default_mask]
        priority_layer[default_mask] = 1000


    # Upscale if requested
    if target_size:
        overlay = cv2.resize(overlay, target_size, interpolation=cv2.INTER_LANCZOS4)
    elif scale_factor != 1:
        h, w = overlay.shape[:2]
        overlay = cv2.resize(
            overlay,
            (w * scale_factor, h * scale_factor),
            interpolation=cv2.INTER_LANCZOS4
        )

    # Save
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    Image.fromarray(overlay_rgb).save(output_path, quality=100, subsampling=0)
    print(f"Overlay saved to {output_path}")


# Example usage
overlay_frames(
    # r"C:\Users\toazb\Documents\GitHub\race_simulation\videos\edge_cases\Collision Edge Cases.mp4",
    r"C:\Users\toazb\Documents\GitHub\race_simulation\videos\edge_cases\collision_scalar_fail.mp4",

    "videos/overlay_schematic_colors.png",
    color_priority=["green", "blue", 'yellow'],
    scale_factor=1
)
