import gymnasium as gym
import tensorflow as tf
import numpy as np
import cv2
import os
import imageio

# --- Load Latest Model ---
models_path = "models"
latest_model = sorted(os.listdir(models_path))[-1]
model_path = os.path.join(models_path, latest_model)
print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# --- Preprocessing ---
def preprocess_obs(obs):
    obs = obs.astype(np.float32) / 255.0
    return obs.reshape(1, 96, 96, 3)

# --- Run Simulation & Save GIF ---
def play_and_record_gif(env, model, n_episodes=1, gif_path="car_racing.gif"):
    all_frames = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            obs_processed = preprocess_obs(obs)
            action_probs = model.predict(obs_processed, verbose=0)[0]
            action = int(np.argmax(action_probs))

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            frame = env.render()  # RGB frame
            all_frames.append(frame)

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()

    print(f"Saving GIF with {len(all_frames)} frames to: {gif_path}")
    imageio.mimsave(gif_path, all_frames, fps=30)
    print("GIF saved.")

# --- Setup Env & Run ---
seed = 42
env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=False
)
env.reset(seed=seed)
np.random.seed(seed)
tf.random.set_seed(seed)

play_and_record_gif(env, model, n_episodes=1)
