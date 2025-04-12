import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create env
env = gym.make("CarRacing-v3", render_mode="rgb_array")
obs, _ = env.reset()

# Take one dummy step to trigger rendering
action = np.array([0.0, 0, 0.0], dtype=np.float32)  # Straight + gas
obs, _, _, _, _ = env.step(action)

# Render frame
for _ in range(100):
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

frame = env.render()

# Now frame should have visual data (not zeros)
print("Frame mean value:", np.mean(frame))  # Debug value check

# Crop and process
# cropped = frame
# cropped = frame[0:75, 0:75]


def green_mask(observation):
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    green = np.zeros_like(observation, np.uint8)
    green[mask_green > 0] = observation[mask_green > 0]
    return green

def gray_scale(observation):
    return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

def blur_image(observation):
    return cv2.GaussianBlur(observation, (5, 5), 0)

def canny_edge_detector(observation):
    return cv2.Canny(observation, 50, 150)

green = green_mask(frame)
gray = gray_scale(green)
blur = blur_image(gray)
canny = canny_edge_detector(blur)

# Define dynamic crop dimensions
y_start, y_end = 200, 205
x_start, x_end = 200, 400

# Apply cropping
cropped = canny[y_start:y_end, x_start:x_end]

# Draw red rectangle using dynamic values
frame_with_rect = frame.copy()
cv2.rectangle(
    frame_with_rect,
    (x_start, y_start),
    (x_end, y_end),
    (255, 0, 0), 2  # Red box in BGR
)


# Plot steps
# Draw red rectangle on original frame to indicate the crop area
fig, axs = plt.subplots(1, 6, figsize=(18, 4))
axs[0].imshow(cv2.cvtColor(frame_with_rect, cv2.COLOR_RGB2BGR))
axs[0].set_title("Original + Crop Box")

axs[1].imshow(cv2.cvtColor(green, cv2.COLOR_RGB2BGR))
axs[1].set_title("Green Masked")
axs[2].imshow(gray, cmap='gray')
axs[2].set_title("Grayscale")
axs[3].imshow(blur, cmap='gray')
axs[3].set_title("Blurred")
axs[4].imshow(canny, cmap='gray')
axs[4].set_title("Canny Edges")
axs[5].imshow(cropped, cmap='gray')
axs[5].set_title("Cropped")

for ax in axs:
    ax.axis("off")
plt.tight_layout()
# plt.show()

plt.savefig("images/image_processing.png")

env.close()
