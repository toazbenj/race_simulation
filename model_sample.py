import gymnasium as gym
import tensorflow as tf
import numpy as np
import cv2  # For visualization

# Initialize the CarRacing-v3 environment in discrete mode
env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

# Get the number of discrete actions
n_actions = env.action_space.n  # Discrete action space

# Define the CNN-based policy network for discrete actions
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(96, 96, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(n_actions, activation="softmax")  # Outputs probabilities for each discrete action
])

# Randomly initialize weights (replace with `model.load_weights()` if using a trained model)
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

            # action_probs = model.predict(obs_processed, verbose=0)[0]  # Predict action probabilities
            # action = int(np.argmax(action_probs))  # Ensure action is an integer

            with tf.GradientTape() as tape:
                probas = model(obs_processed)
                logits = tf.math.log(probas + tf.keras.backend.epsilon())
                action = tf.random.categorical(logits, num_samples=1)

            # Step environment
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # Render frame using OpenCV
            if display:
                frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)  # Convert color format
                frame_resized = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_CUBIC)  # Resize with high clarity

                cv2.imshow("CarRacing-v3 Agent", frame_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                    break

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()

# Run the visualization with trained policy
play_with_policy(env, model, n_episodes=3, display=True)
