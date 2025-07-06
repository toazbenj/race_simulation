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

import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_requests_with_results_3d(json_path, plt_path):
    # Load JSON from file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    coords = data["requests"]
    results = data["results"]
    
    # Extract x, y, z coordinates
    x = [pt[0] for pt in coords]
    y = [pt[1] for pt in coords]
    z = [pt[2] for pt in coords]
    
    # Set colors based on results
    colors = ['red' if res else 'blue' for res in results]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(x, y, z, c=colors, edgecolor='black', s=50)
    
    # Set labels and title
    ax.set_xlabel('Progress Weight')
    ax.set_ylabel('Bounds Weight')
    ax.set_zlabel('Proximity')
    ax.set_title('Sembas Samples - Complete Success')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Failed'),
                      Patch(facecolor='blue', label='Successful')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Improve the view angle
    ax.view_init(elev=20, azim=45)
    
    # Save the plot
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# Main

plot_requests_with_results_3d("data/test.json", "data/sembas_results_3d.png")
# plot_requests_with_results("data/results_passes_100.json", "data/sembas_results.png")
