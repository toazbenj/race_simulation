import gymnasium as gym
import cv2
import numpy as np
import math

# Initialize CarRacing-v3 environment with high-resolution rendering
env = gym.make("CarRacing-v3", render_mode="rgb_array", render_fps=60)


# Improved circular policy: smooth gradual turns
def circular_policy(step):
    steering = 0.2 * math.sin(step * 0.01) - 0.1  # Reduce oscillation and add slight left turn bias
    throttle = 0.6  # Reduce speed to improve control
    return np.array([steering, throttle, 0.0])  # Steering, Throttle, No Brake


def run_circular_policy(env, n_episodes=1):
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            action = circular_policy(step)
            obs, reward, done, truncated, _ = env.step(action)

            # Render frame directly from the environment in high resolution
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_CUBIC)  # Resize with high clarity
            cv2.imshow("CarRacing - High Resolution", frame_resized)  # Display enlarged resolution
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
            step += 1

    env.close()
    cv2.destroyAllWindows()


# Run the improved circular policy
env.reset()
run_circular_policy(env)
