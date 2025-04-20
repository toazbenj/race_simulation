import gymnasium as gym
import tensorflow as tf
import numpy as np
import cv2
import os
import imageio
import imageio
from action_graphing import *


# --- Load Model ---
# models_path = "models"
# latest_model = sorted(os.listdir(models_path))[-1]
# model_path = os.path.join(models_path, latest_model)

model_path = 'good_models/ddqn_long_run.h5'

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path, compile=False)

# --- PID Control ---
def pid(error, prev_error):
    Kp, Ki, Kd = 0.02, 0.03, 0.2
    return Kp * error + Ki * (error + prev_error) + Kd * (error - prev_error)

# --- Image Processing ---
def preprocess_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    green = np.zeros_like(frame)
    green[mask > 0] = frame[mask > 0]
    gray = cv2.cvtColor(green, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    cropped = canny[200:205, 200:400]
    return cropped

def find_error(canny_crop, prev_error):
    nz = cv2.findNonZero(canny_crop)
    mid = 100
    if nz is None:
        return prev_error
    if nz[:, 0, 0].max() == nz[:, 0, 0].min():
        return -15 if nz[:, 0, 0].max() >= mid else 15
    return ((nz[:, 0, 0].max() + nz[:, 0, 0].min()) / 2) - mid

def preprocess_inputs(crop, error):
    crop = crop.astype(np.float32) / 255.0
    crop = crop.reshape(1, 5, 200, 1)
    return [crop, np.array([[error / 100]], dtype=np.float32)]


# --- Main ---
seed = 3
env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True
)
env.reset(seed=seed)
np.random.seed(seed)
tf.random.set_seed(seed)

gas_levels = [0.0, 0.5, 1.0]
frames = []
n_steps = 100
action_lst = np.zeros((n_steps, 3))
reward_lst = []

obs, _ = env.reset()
total_reward = 0
done = False
prev_error = 0

for step in range(n_steps):
    frame = env.render()  # Capture RGB frame
    crop = preprocess_frame(frame)
    error = find_error(crop, prev_error)

    steering = pid(error, prev_error)
    q_values = model.predict(preprocess_inputs(crop, error), verbose=0)[0]
    action_index = np.argmax(q_values)
    gas = gas_levels[action_index]

    action = np.array([steering, gas, 0], dtype=np.float32)
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    prev_error = error

    action_lst[step, :] = action
    reward_lst.append(total_reward)
    frames.append(frame.copy())

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_resized = cv2.resize(frame_bgr, (400, 400))
    cv2.imshow("Hybrid PID+DQN Agent", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        env.close()
        cv2.destroyAllWindows()

env.close()
gif_path = 'video/ddqn.gif'
print(f"Saving GIF with {len(frames)} frames to: {gif_path}")
imageio.mimsave(gif_path, frames[1:], fps=30)
print("GIF saved.")

graph_actions(action_lst, 'ddqn')
graph_reward(reward_lst, 'ddqn')
print(f"Simulation complete. Total reward: {reward_lst[-1]:.2f}")