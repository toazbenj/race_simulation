import gymnasium as gym
import tensorflow as tf
import numpy as np
import cv2
from collections import deque
from datetime import datetime
from pathlib import Path
import os
import keras
from tensorflow.summary import create_file_writer

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
    x1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(image_input)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU(alpha=0.1)(x1)
    x1 = tf.keras.layers.MaxPooling2D()(x1)
    x1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.LeakyReLU(alpha=0.1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dropout(0.3)(x1)

    error_input = tf.keras.Input(shape=(1,), name="error")
    x2 = tf.keras.layers.Dense(16, activation="relu")(error_input)
    x2 = tf.keras.layers.LeakyReLU(alpha=0.1)(x2)

    concat = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(64, activation="relu")(concat)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # binary output

    model = tf.keras.models.Model(inputs=[image_input, error_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

def preprocess_inputs(crop, error):
    crop = crop.astype(np.float32) / 255.0
    crop = crop.reshape(1, 5, 200, 1)
    return [crop, np.array([[error/100]], dtype=np.float32)]

def epsilon_policy(crop, error, epsilon):
    if np.random.rand() < epsilon:
        return np.random.uniform(0, 1)
    else:
        proba =  float(model.predict(preprocess_inputs(crop, error), verbose=0)[0][0])
        # print("proba: ", proba)
        return proba

def get_epsilon(ep, min_epsilon=0.05, decay_rate=0.03):
    return max(min_epsilon, np.exp(-decay_rate * ep))

def train_step():
    batch = [replay_buffer[np.random.randint(len(replay_buffer))] for _ in range(batch_size)]
    crops, errors, speeds = zip(*batch)
    X_img = np.array(crops).reshape(batch_size, 5, 200, 1)
    X_err = np.array(errors).reshape(batch_size, 1)

    target_preds = target_model.predict([X_img, X_err], verbose=0)
    loss = model.train_on_batch([X_img, X_err], target_preds)

    with writer.as_default():
        tf.summary.scalar("Loss/train", loss, step=ep)


# --- Setup ---
model = build_model()
target_model = tf.keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = create_file_writer(log_dir)

replay_buffer = deque(maxlen=2000)
loss_fn = tf.keras.losses.MeanSquaredError()

env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
np.random.seed(42)
tf.random.set_seed(42)
env.reset(seed=42)

episodes = 50
batch_size = 64
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
        probability = epsilon_policy(crop, error, epsilon)
        gas = int(probability >= 0.5)

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

            with writer.as_default():
                for layer in model.layers:
                    weights = layer.get_weights()
                    if weights:
                        tf.summary.histogram(f"{layer.name}/weights", weights[0], step=ep)

        if terminated or truncated:
            break

        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame_resized = cv2.resize(frame_bgr, (400, 400))
        # cv2.imshow("Hybrid PID+DQN Agent", frame_resized)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     env.close()
        #     cv2.destroyAllWindows()

    print(f"Episode {ep+1} - Reward: {total_reward:.2f}")
    rewards.append(total_reward)

    with writer.as_default():
        tf.summary.scalar("Reward/Episode", total_reward, step=ep)

    if total_reward > best_score:
        best_score = total_reward
        best_weights = model.get_weights()
        print(f" - New best score: {best_score:.2f}")
        if best_score > 300:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            Path("models").mkdir(parents=True, exist_ok=True)
            file_path = os.path.join("models", f"{timestamp}_upgraded_hybrid_dqn.h5")
            keras.saving.save_model(model, file_path)

            # for layer in model.layers:
            #     weights = layer.get_weights()
            #     if weights:
            #         print(f"Layer: {layer.name}")
            #         for i, w in enumerate(weights):
            #             print(f"  Weight {i}: shape={w.shape}")
            #             print(w)  # or use print(w[:5]) to print the first few values

    if ep % 5 == 0:
        target_model.set_weights(model.get_weights())

env.close()
writer.close()
