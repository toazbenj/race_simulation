import gymnasium as gym
import tensorflow as tf
import numpy as np
import cv2  # For visualization
import os

# Load the latest trained model
models_path = "models"
latest_model = sorted(os.listdir(models_path))[-1]  # Get latest model file
model_path = os.path.join(models_path, latest_model)

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)  # Load the trained model

# Initialize the CarRacing-v3 environment in discrete mode
env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

# Function to preprocess observation (resize, normalize)
def preprocess_obs(obs):
    obs = obs.astype(np.float32) / 255.0  # Normalize pixel values
    return obs.reshape(1, 96, 96, 3)

# Function to play the game using the trained policy
def play_with_policy(env, model, n_episodes=5, display=True):
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Get model prediction
            obs_processed = preprocess_obs(obs)
            action_probs = model.predict(obs_processed, verbose=0)[0]  # Predict Q-values
            action = int(np.argmax(action_probs))  # Select action with highest Q-value

            # Step environment
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # Render frame using OpenCV
            if display:
                frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)  # Convert color format
                frame_resized = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_CUBIC)  # Resize for clarity
                cv2.imshow("CarRacing-v3 Agent", frame_resized)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                    env.close()
                    cv2.destroyAllWindows()
                    return

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()

# Run the visualization with trained policy
play_with_policy(env, model, n_episodes=3, display=True)
