import gymnasium as gym
import tensorflow as tf
import numpy as np
import cv2  # For visualization

# Initialize the CarRacing-v2 environment
env = gym.make("CarRacing-v3", render_mode="rgb_array")

# Set input and output dimensions
n_inputs = (96, 96, 3)  # Image input
n_outputs = env.action_space.shape[0]  # Continuous 3D action space

# Define the trained CNN-based policy network
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=n_inputs),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(n_outputs, activation="tanh")  # Outputs continuous values
])

# Load trained weights (ensure you have saved weights after training)
# model.load_weights("trained_carracing_policy.h5")

# Instead, initialize the model randomly
model.set_weights([np.random.randn(*w.shape) for w in model.get_weights()])

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
            action = model.predict(obs_processed, verbose=0)[0]  # Predict continuous actions

            # Step environment
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # Render frame using OpenCV
            if display:
                frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)  # Convert color format
                frame_resized = cv2.resize(frame, (300, 300),
                                           interpolation=cv2.INTER_CUBIC)  # Resize with high clarity

                cv2.imshow("CarRacing-v2 Agent", frame_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                    break

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()

# Run the visualization with trained policy
play_with_policy(env, model, n_episodes=3, display=True)
