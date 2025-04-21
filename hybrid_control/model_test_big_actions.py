import gymnasium as gym
import tensorflow as tf
import numpy as np
import cv2
import os
from action_graphing import *
import imageio

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

def preprocess_inputs(crop, error):
    crop = crop.astype(np.float32) / 255.0
    crop = crop.reshape(1, 5, 200, 1)
    return [crop, np.array([[error/100]], dtype=np.float32)]

# --- Run policy ---
def play_with_policy(env, model, n_episodes=3, display=True):


    for episode in range(n_episodes):
        obs, _ = env.reset(seed=3)
        total_reward = 0
        done = False
        prev_error = 0
        steps = 600
        action_lst = np.zeros((steps, 3))
        reward_lst = []
        frames = []

        for step in range(steps):
            frame = env.render()
            crop = preprocess_frame(frame)
            error = find_error(crop, prev_error)
            steering = pid(error, prev_error)

            probability = model.predict(preprocess_inputs(crop, error), verbose=0)
            idx = np.argmax(probability)
            print(probability)
            pick = [(1,0.2), (1,0), (0.5,0), (0.1, 0.2), (0.1, 0)][idx]

            action = np.array([steering, *pick], dtype=np.float32)
            action_lst[step, :] = action

            obs, reward, done, truncated, _ = env.step(action)
            prev_error = error
            total_reward += reward
            reward_lst.append(total_reward)

            frames.append(frame.copy())

            if display:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame_bgr, (400, 400))
                cv2.imshow("Hybrid PID+DQN Agent", frame_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    env.close()
                    cv2.destroyAllWindows()
                    return

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()

    return action_lst, reward_lst, frames

if __name__ == "__main__":
    # --- Load model ---
    # models_path = "good_models"
    # latest_model = sorted(os.listdir(models_path))[-1]
    # model_path = os.path.join(models_path, latest_model)

    model_path = 'good_models/big_action_deep_dqn_20250419_1550_hybrid_dqn_pid.h5'

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    seed = 3
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    env.reset(seed=seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    action_lst, reward_lst, frames = play_with_policy(env, model, n_episodes=1, display=True)

    graph_actions(action_lst, 'last_chance_ddqn')
    graph_reward(reward_lst, 'last_chance_ddqn')
    print(f"Simulation complete. Total reward: {reward_lst[-1]:.2f}")

    name = "last_chance_ddqn.gif"
    imageio.mimsave("images/"+name, frames, fps=15)
    print("GIF saved as "+ name)
