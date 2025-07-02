import json
import matplotlib.pyplot as plt

def plot_requests_with_results(json_path, plt_path):
    # Load JSON from file
    with open(json_path, 'r') as f:
        data = json.load(f)

    coords = data["requests"]
    results = data["results"]

    x = [pt[0] for pt in coords]
    y = [pt[1] for pt in coords]
    colors = ['red' if res else 'blue' for res in results]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=colors, edgecolor='black')
    plt.title("Sembas Samples - Successful Passes")
    plt.xlabel("Bounds Weight")
    plt.ylabel("Proximity Weight")
    plt.grid(False)
    # plt.show()

    plt.savefig(plt_path)

# Example usage
plot_requests_with_results("data/results_passes_100.json", "data/sembas_results.png")
