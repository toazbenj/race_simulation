import gymnasium as gym
import tensorflow as tf
import numpy as np
import cv2
from collections import deque
from datetime import datetime
from pathlib import Path
import os
import keras

# --- PID Controller ---
def pid(error, prev_error):
    Kp, Ki, Kd = 0.02, 0.03, 0.2
    return Kp * error + Ki * (error + prev_error) + Kd * (error - prev_error)

# --- Vision Processing ---
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

def find_error(crop, prev_error):
    nz = cv2.findNonZero(crop)
    mid = 100
    if nz is None:
        return prev_error
    if nz[:, 0, 0].max() == nz[:, 0, 0].min():
        return -15 if nz[:, 0, 0].max() >= mid else 15
    return ((nz[:, 0, 0].max() + nz[:, 0, 0].min()) / 2) - mid

# --- Model ---
def build_model():
    image_input = tf.keras.Input(shape=(5, 200, 1), name="canny_crop")
    x1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(image_input)
    x1 = tf.keras.layers.MaxPooling2D()(x1)
    x1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dropout(0.3)(x1)

    error_input = tf.keras.Input(shape=(1,), name="error")
    x2 = tf.keras.layers.Dense(16, activation="relu")(error_input)

    concat = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(64, activation="relu")(concat)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=[image_input, error_input], outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model

def preprocess_inputs(crop, error):
    crop = crop.astype(np.float32) / 255.0
    crop = crop.reshape(1, 5, 200, 1)
    return [crop, np.array([[error]], dtype=np.float32)]

def epsilon_policy(crop, error, epsilon):
    if np.random.rand() < epsilon:
        return np.random.uniform(0, 1)
    else:
        return float(model.predict(preprocess_inputs(crop, error), verbose=0)[0][0])

def get_epsilon(ep, min_epsilon=0.05, decay_rate=0.03):
    return max(min_epsilon, np.exp(-decay_rate * ep))

def train_step():
    batch = [replay_buffer[np.random.randint(len(replay_buffer))] for _ in range(batch_size)]
    crops, errors, speeds = zip(*batch)
    X_img = np.array(crops).reshape(batch_size, 5, 200, 1)
    X_err = np.array(errors).reshape(batch_size, 1)/100 # normalized

    target_preds = target_model.predict([X_img, X_err], verbose=0)
    model.train_on_batch([X_img, X_err], target_preds)

# --- Setup ---
model = build_model()
target_model = tf.keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

replay_buffer = deque(maxlen=2000)
loss_fn = tf.keras.losses.MeanSquaredError()

env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
np.random.seed(42)
tf.random.set_seed(42)
env.reset(seed=42)

episodes = 100
batch_size = 32
rewards = []
best_score = -1000

# --- Training ---
for ep in range(episodes):
    obs, _ = env.reset()
    total_reward = 0
    prev_error = 0
    speed = 0
    smoothed_reward = 0

    for step in range(1000):
        frame = env.render()
        crop = preprocess_frame(frame)
        error = find_error(crop, prev_error)

        epsilon = get_epsilon(ep)
        gas = epsilon_policy(crop, error, epsilon)
        steering = pid(error, prev_error)
        brake = 0.0

        action = np.array([steering, gas, brake], dtype=np.float32)

        obs, reward, terminated, truncated, _ = env.step(action)
        smoothed_reward = 0.9 * smoothed_reward + 0.1 * float(reward)
        total_reward += smoothed_reward

        replay_buffer.append((crop, error, gas))
        prev_error = error

        if len(replay_buffer) > 1000 and step % 4 == 0:
            train_step()

        if terminated or truncated:
            break

    print(f"Episode {ep+1} - Reward: {total_reward:.2f}")
    rewards.append(total_reward)

    if total_reward > best_score:
        best_score = total_reward
        best_weights = model.get_weights()
        print(f" - New best score: {best_score:.2f}")
        if best_score > 100:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            Path("models").mkdir(parents=True, exist_ok=True)
            file_path = os.path.join("models", f"{timestamp}_upgraded_hybrid_dqn.h5")
            keras.saving.save_model(model, file_path)

    if ep % 5 == 0:
        target_model.set_weights(model.get_weights())

env.close()
