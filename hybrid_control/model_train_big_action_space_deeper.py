import gymnasium as gym
import tensorflow as tf
import numpy as np
import cv2
from collections import deque
from datetime import datetime
import os
import keras
import matplotlib.pyplot as plt
from pathlib import Path

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

def find_error(canny_crop, prev_error):
    nz = cv2.findNonZero(canny_crop)
    mid = 100
    if nz is None:
        return prev_error
    if nz[:, 0, 0].max() == nz[:, 0, 0].min():
        return -15 if nz[:, 0, 0].max() >= mid else 15
    return ((nz[:, 0, 0].max() + nz[:, 0, 0].min()) / 2) - mid


def build_model():
    image_input = tf.keras.Input(shape=(5, 200, 1), name="canny_crop")

    x1 = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(image_input)
    x1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x1)
    x1 = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x1)
    x1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x1)
    x1 = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x1)
    x1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x1)
    x1 = tf.keras.layers.Flatten()(x1)

    # Simpler error processing
    error_input = tf.keras.Input(shape=(1,), name="error")
    x2 = tf.keras.layers.Dense(8, activation="relu")(error_input)

    # Fuse and output Q-values for 3 gas levels
    concat = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(16, activation="relu")(concat)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    output = tf.keras.layers.Dense(5, activation=None)(x)

    model = tf.keras.models.Model(inputs=[image_input, error_input], outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model

def preprocess_inputs(crop, error):
    crop = crop.astype(np.float32) / 255.0
    crop = crop.reshape(1, 5, 200, 1)
    return [crop, np.array([[error/100]], dtype=np.float32)]

def epsilon_policy(crop, error, epsilon):
    if np.random.rand() < epsilon:
        random_values = np.random.rand(5)
        normalized_array = random_values / random_values.sum()
        normalized_array = normalized_array.reshape(1, 5)
        return normalized_array
    else:
        return model.predict(preprocess_inputs(crop, error), verbose=0)

def train_step(ep, action_lst):
    batch = [replay_buffer[np.random.randint(len(replay_buffer))] for _ in range(batch_size)]
    crops, errors, actions, rewards = zip(*batch)
    X_img = np.array(crops).reshape(batch_size, 5, 200, 1)
    X_err = np.array(errors).reshape(batch_size, 1)
    y = np.zeros((batch_size, 5))
    for i in range(batch_size):
        action_idx = action_lst.index(actions[i])  # get correct class (0, 1, 2)
        y[i, action_idx] = rewards[i]
    loss = model.train_on_batch([X_img, X_err], y)

    with writer.as_default():
        tf.summary.scalar("Loss/train", loss, step=ep)

    return loss

# --- Setup ---
model = build_model()
replay_buffer = deque(maxlen=5000)
loss_fn = tf.keras.losses.MeanSquaredError()

env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
seed = 3
np.random.seed(seed)
tf.random.set_seed(seed)
env.reset(seed=seed)

# --- Training ---
episodes = 20
batch_size = 64
rewards = []
best_score = -1000
gas_lst = [0, 0.5, 1]
losses = []

# Create a log directory
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# --- Setup ---
model = build_model()
replay_buffer = deque(maxlen=5000)
loss_fn = tf.keras.losses.MeanSquaredError()

env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
seed = 3
np.random.seed(seed)
tf.random.set_seed(seed)
env.reset(seed=seed)

# --- Training ---
episodes = 20
batch_size = 64
rewards = []
best_score = -1000

action_lst = [(1,0.2), (1,0), (0.5,0), (0.1, 0.2), (0.1, 0)]

for ep in range(episodes):
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    prev_error = 0
    speed = 0

    for step in range(1000):
        frame = env.render()
        crop = preprocess_frame(frame)
        error = find_error(crop, prev_error)

        steering = pid(error, prev_error)
        probability = epsilon_policy(crop, error, epsilon=max(0.1, 1 - ep/episodes))
        idx = np.argmax(probability)

        pick = action_lst[idx]
        action = np.array([steering, *pick], dtype=np.float32)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Store experience
        replay_buffer.append((crop, error, pick, reward))
        prev_error = error

        if terminated or truncated:
            break

    print(f"Episode {ep+1} - Reward: {total_reward:.2f}")
    rewards.append(total_reward)

    with writer.as_default():
        tf.summary.scalar("Reward/Episode", total_reward, step=ep)

    if total_reward > best_score:
        best_weights = model.get_weights()
        best_score = total_reward
        print(f" - New best score! {best_score:.2f}")

        if best_score >  300:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            file_path = os.path.join("models", f"{timestamp}_hybrid_dqn_pid.h5")
            keras.saving.save_model(model, file_path)

    if ep > 5 and len(replay_buffer) > batch_size:
        loss = train_step(ep, action_lst)
        losses.append(loss)

        # Log Conv2D layer weights as histograms
        with writer.as_default():
            for layer in model.layers:
                weights = layer.get_weights()
                if weights:
                    tf.summary.histogram(f"{layer.name}/weights", weights[0], step=ep)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
file_path = os.path.join("models", f"{timestamp}_hybrid_dqn_pid.h5")
keras.saving.save_model(model, file_path)

env.close()
writer.close()

# ========== Plot and Save Curves ==========
Path("images").mkdir(exist_ok=True)
model_name = Path(log_dir).name
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Reward Plot
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Reward per Episode")
plt.plot([max(rewards[:i+1]) for i in range(len(rewards))], linestyle='--', label="Best Reward Overall")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(f"Reward Curve – {model_name}")
plt.grid(True)
plt.legend()
reward_path = f"images/{model_name}_reward_plot_{timestamp}.png"
plt.savefig(reward_path)
plt.show()

# Loss Plot
if losses:
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss per Train Step")
    plt.xlabel("Train Step")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve – {model_name}")
    plt.grid(True)
    plt.legend()
    loss_path = f"images/{model_name}_loss_plot_{timestamp}.png"
    plt.savefig(loss_path)
    plt.show()

print(f"Saved plots to: {reward_path}")
