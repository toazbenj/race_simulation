import gymnasium as gym
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # ✅ Must be BEFORE importing pyplot
import matplotlib.pyplot as plt
import imageio

# --- Image Processing ---
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

def find_error(observation, previous_error):
    green = green_mask(observation)
    grey = gray_scale(green)
    blur = blur_image(grey)
    canny = canny_edge_detector(blur)
    cropped = canny[200:250, 200:400]

    nz = cv2.findNonZero(cropped)
    mid = 100

    cropped = cv2.resize(cropped, (observation.shape[1], cropped.shape[0]), interpolation=cv2.INTER_NEAREST)

    if nz is None:
        return previous_error, canny, cropped

    if nz[:, 0, 0].max() == nz[:, 0, 0].min():
        if 20 < nz[:, 0, 0].max() < 30:
            return previous_error, canny, cropped
        return (-15 if nz[:, 0, 0].max() >= mid else 15), canny, cropped
    else:
        err = ((nz[:, 0, 0].max() + nz[:, 0, 0].min()) / 2) - mid
        return err, canny, cropped

def pid(error, previous_error):
    Kp = 0.02
    Ki = 0.03
    Kd = 0.2
    return Kp * error + Ki * (error + previous_error) + Kd * (error - previous_error)

# --- Setup Environment ---
env = gym.make("CarRacing-v3", render_mode="rgb_array")
env.reset(seed=42)
obs, _ = env.reset()
previous_error = 0
speed = 0
total_reward = 0
min_speed = 0.2

# --- Prepare GIF frames ---
frames = []

# --- Setup Matplotlib (Agg backend) ---
fig, axs = plt.subplots(3, 1, figsize=(4, 8))

# --- Main Loop ---
for step in range(6000):  # Adjust frame count
    print(step)
    frame = env.render()

    error, canny, cropped = find_error(frame, previous_error)
    steering = pid(error, previous_error)

    # Control logic
    if abs(error) < 5:
        gas = 0.75
        brake = 0
    elif speed > 0.3:
        gas = 0
        brake = 0.1
    elif speed < min_speed:
        gas = 0.3
        brake = 0
    else:
        gas = 0.1
        brake = 0

    speed = max(0.0, speed + gas - brake)
    action = np.array([steering, gas, brake], dtype=np.float32)

    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    previous_error = error

    # Visualization
    imgs = [frame, canny, cropped]
    titles = ["Original", "Canny", "Cropped"]
    cmaps = [None, "gray", "gray"]

    for i in range(3):
        axs[i].clear()
        if cmaps[i]:
            axs[i].imshow(imgs[i], cmap=cmaps[i])
        else:
            axs[i].imshow(cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))
        axs[i].set_title(titles[i])
        axs[i].axis("off")
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.tight_layout()
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape((h, w, 4))  # 4 channels = RGBA
    frames.append(image.copy())

    if terminated or truncated:
        break

# --- Cleanup ---
env.close()
plt.close()

# --- Save GIF ---
imageio.mimsave("images/pid_canvas_output.gif", frames, fps=15)
print(f"Simulation complete. Total reward: {total_reward:.2f}")
print("GIF saved as 'pid_canvas_output.gif'")
