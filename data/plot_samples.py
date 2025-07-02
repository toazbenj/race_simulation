import json
import matplotlib.pyplot as plt

def plot_requests_with_results(json_path):
    # Load JSON from file
    with open(json_path, 'r') as f:
        data = json.load(f)

    coords = data["requests"]
    results = data["results"]

    x = [pt[0] for pt in coords]
    y = [pt[1] for pt in coords]
    colors = ['blue' if res else 'red' for res in results]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=colors, edgecolor='black')
    plt.title("Sembas Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

# Example usage
plot_requests_with_results("data/results.json")
