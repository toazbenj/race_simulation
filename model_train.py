import gymnasium as gym
import tensorflow as tf
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from pathlib import Path

# Set up directories for saving images
IMAGES_PATH = Path() / "images" / "rl"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Function to preprocess images (convert to grayscale & resize)
def preprocess_obs(obs):
    obs = obs.astype(np.float32) / 255.0  # Normalize pixels
    return obs.reshape(1, 96, 96, 3)  # Keep batch dimension

# Epsilon-greedy policy for action selection
def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)  # Random action
    else:
        Q_values = model.predict(preprocess_obs(state), verbose=0)[0]  # Predict Q-values
        return int(np.argmax(Q_values))  # Choose best action (ensure it's an int)

# Sample a batch of experiences from replay buffer
def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(6)
    ]  # [states, actions, rewards, next_states, dones, truncateds]

# Play one step and store in replay buffer
def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, truncated, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done, truncated))
    return next_state, reward, done, truncated, info

# Training function
def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones, truncateds = experiences

    next_Q_values = model(next_states)  # No preprocessing needed for batch
    max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)

    runs = 1.0 - (dones | truncateds)  # Episode is not done or truncated
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = tf.reshape(target_Q_values, (-1, 1))


    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)  # No preprocessing needed
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# Initialize the environment
env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
input_shape = (96, 96, 3)
n_outputs = env.action_space.n  # 5 discrete actions

# 🛠️ **Fixed Neural Network Model (Using CNN)**
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(n_outputs, activation="softmax")  # Output probabilities for discrete actions
])

# Set up replay buffer
replay_buffer = deque(maxlen=2000)

# Set seeds for reproducibility
env.reset(seed=42)
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
rewards = []
best_score = 0
batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
loss_fn = tf.keras.losses.MeanSquaredError()

# 🏁 **Training Loop**
for episode in range(20):
    obs, info = env.reset()
    for step in range(100):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, truncated, info = play_one_step(env, obs, epsilon)
        if done or truncated:
            break

    print(f"\rEpisode: {episode + 1}, Steps: {step + 1}, eps: {epsilon:.3f}", end="")
    rewards.append(step)

    if step >= best_score:
        best_weights = model.get_weights()
        best_score = step

    if episode > 10:
        training_step(batch_size)

model.set_weights(best_weights)  # Restore best model weights

# 🎯 **Plot Training Progress**
plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.grid(True)
save_fig("dqn_rewards_plot")
plt.show()
