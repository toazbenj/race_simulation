import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from action_graphing import *
matplotlib.use('TkAgg')  # For Windows/Mac/Linux GUI support

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
    cropped = canny[200:210, :]

    nz = cv2.findNonZero(cropped)
    mid = 300

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


# main

# --- Setup Environment ---
env = gym.make("CarRacing-v3", render_mode="rgb_array")
obs, _ = env.reset()
previous_error = 0
speed = 0
total_reward = 0
min_speed = 0.2
steps = 100
action_lst = np.zeros((steps,3))
reward_lst = []

# --- Setup Matplotlib Live View ---
fig, axs = plt.subplots(3, 1, figsize=(4, 8))
plt.ion()
plt.show(block=False)

for step in range(steps):
    frame = env.render()

    error, canny, cropped = find_error(frame, previous_error)
    steering = pid(error, previous_error)

    # Adjust throttle logic to maintain motion
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

    # naive policy
    # gas = 1.5 * step % 2
    # brake = 0

    speed = max(0.0, speed + gas - brake)
    action = np.array([steering, gas, brake], dtype=np.float32)
    action_lst[step,:] = action

    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    previous_error = error
    reward_lst.append(total_reward)

    # OpenCV Game View
    # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow("CarRacing-v3 PID", cv2.resize(frame_bgr, (400, 400)))

    # Matplotlib live update of processing stages
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

    print(step)

plt.ioff()
env.close()
cv2.destroyAllWindows()
plt.close()

graph_actions(action_lst, 'pid')
graph_reward(reward_lst, 'pid')
print(f"Simulation complete. Total reward: {total_reward:.2f}")
