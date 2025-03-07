import gymnasium as gym
import tensorflow as tf
import numpy as np
import cv2  # For visualization

# Initialize the CarRacing-v3 environment
env = gym.make("CarRacing-v3", render_mode="rgb_array")
n_inputs = (96, 96, 3)  # Image input
n_outputs = env.action_space.shape[0]  # Continuous 3D action space

def create_policy_network():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=n_inputs),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(n_outputs, activation="tanh")
    ])
    return model

policy_network = create_policy_network()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def preprocess_obs(obs):
    return obs.astype(np.float32) / 255.0  # Normalize pixels

def play_one_step(env, obs, model):
    with tf.GradientTape() as tape:
        obs_processed = preprocess_obs(obs).reshape(1, 96, 96, 3)
        action = model(obs_processed, training=True)[0]
        log_prob = tf.reduce_sum(tf.math.log(tf.clip_by_value(action, 1e-9, 1.0)))
    obs, reward, done, truncated, _ = env.step(action.numpy())
    return obs, reward, done, truncated, tape, log_prob

def play_multiple_episodes(env, model, n_episodes=10, n_max_steps=1000):
    all_rewards = []
    all_log_probs = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_rewards = []
        episode_log_probs = []
        for step in range(n_max_steps):
            obs, reward, done, truncated, tape, log_prob = play_one_step(env, obs, model)
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            if done or truncated:
                break
        all_rewards.append(episode_rewards)
        all_log_probs.append(episode_log_probs)
    return all_rewards, all_log_probs

def compute_loss(all_rewards, all_log_probs, discount_factor=0.99):
    discounted_rewards = []
    for rewards in all_rewards:
        discounted = []
        sum_reward = 0
        for r in reversed(rewards):
            sum_reward = r + discount_factor * sum_reward
            discounted.insert(0, sum_reward)
        discounted_rewards.append(discounted)
    discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
    all_log_probs = tf.convert_to_tensor(all_log_probs, dtype=tf.float32)
    loss = -tf.reduce_mean(discounted_rewards * all_log_probs)
    return loss

def train_policy(env, model, optimizer, n_iterations=100, n_episodes=10):
    print('training')
    for iteration in range(n_iterations):
        all_rewards, all_log_probs = play_multiple_episodes(env, model, n_episodes)
        with tf.GradientTape() as tape:
            loss = compute_loss(all_rewards, all_log_probs)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Iteration {iteration+1}, Mean Reward: {np.mean([sum(r) for r in all_rewards]):.2f}")

train_policy(env, policy_network, optimizer, n_iterations=50)
