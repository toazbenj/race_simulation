import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay

# --- Load Model ---
models_path = "models"
latest_model = sorted(os.listdir(models_path))[-1]
model_path = os.path.join(models_path, latest_model)
print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path, compile=False)

# --- Video Utils ---
def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data=f'''
        <video alt="simulation" autoplay loop controls style="height: 400px;">
            <source src="data:video/mp4;base64,{encoded.decode('ascii')}" type="video/mp4" />
        </video>'''))
    else:
        print("❌ Could not find video")

def wrap_env(env):
    if not os.path.exists('video'):
        os.makedirs('video')
    return RecordVideo(env, video_folder='video', episode_trigger=lambda ep: True)

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

# --- Run Agent ---
def play_with_policy(env, model, n_episodes=1, display=False):
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        prev_error = 0

        while not done:
            frame = obs
            crop = preprocess_frame(frame)
            error = find_error(crop, prev_error)

            steering = pid(error, prev_error)

            probability = float(model.predict(preprocess_inputs(crop, error), verbose=0))
            gas = int(probability >= 0.5)
            brake = 0.0

            action = np.array([steering, gas, brake], dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            prev_error = error
            total_reward += reward
            done = terminated or truncated

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

# --- Init Env ---
seed = 42
env = wrap_env(gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",
    lap_complete_percent=0.95,
    domain_randomize=False,
    continuous=True
))
env.reset(seed=seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# --- Run & Show ---
play_with_policy(env, model, n_episodes=1, display=False)
show_video()
