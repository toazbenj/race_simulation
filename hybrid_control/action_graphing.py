import numpy as np
import matplotlib.pyplot as plt

def graph_actions(action_lst, model_name):
    action_lst[:, 0] = action_lst[:, 0] / max(np.max(action_lst[:, 0]), np.abs(np.min(action_lst[:, 0])))

    steering = action_lst[:,0]
    gas = action_lst[:,1]
    brake = action_lst[:,2]

    plt.figure(figsize=(10, 5))
    plt.plot(steering, label="Steering Level")
    plt.plot(gas, label="Gas Level")
    plt.plot(brake, label="Brake Level")

    plt.xlabel("Frame")
    plt.ylabel("Action Level")
    plt.title(f"Actions Over Time")
    plt.grid(True)
    plt.legend()
    path = f"images/{model_name}_actions.png"
    plt.savefig(path)
    plt.show()


def graph_reward(reward_lst, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_lst)

    plt.xlabel("Frame")
    plt.ylabel("Reward")
    plt.title(f"Reward Over Time")
    plt.grid(True)
    plt.legend()
    path = f"images/{model_name}_reward.png"
    plt.savefig(path)
    plt.show()

# Main
if __name__ == "__main__":
    action_lst = np.array([[1,1,1],[0,1,0],[1,-1,-1]])
    reward_lst = 10*[100]

    graph_actions(action_lst, "test")
    graph_reward(reward_lst, "test")