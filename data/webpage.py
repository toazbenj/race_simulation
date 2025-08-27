
# Interactive 3D Plotting for Sembas Results
# Fixed Version

import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Enable interactive plots in Jupyter
# %matplotlib widget


# Interactive Matplotlib Version
def plot_requests_with_results_3d_interactive(json_path=None, data=None):
    """
    Interactive 3D scatter plot using matplotlib
    """
    if data is None:
        with open(json_path, 'r') as f:
            data = json.load(f)
    
    coords = data["requests"]
    results = data["results"]
    
    # Extract coordinates
    x = [pt[0] for pt in coords]
    y = [pt[1] for pt in coords]
    z = [pt[2] for pt in coords]
    
    # Set colors
    colors = ['red' if not res else 'blue' for res in results]
    
    # Create interactive 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    scatter = ax.scatter(x, y, z, c=colors, edgecolor='black', s=80, alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('Progress Weight', fontsize=12)
    ax.set_ylabel('Bounds Weight', fontsize=12)
    ax.set_zlabel('Proximity Weight', fontsize=12)
    ax.set_title('Sembas Samples - Successful Races', fontsize=14)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Failed'),
                      Patch(facecolor='blue', label='Successful')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Enable interactive rotation
    ax.mouse_init()
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

# Plotly Version (More Interactive) - FIXED
def plot_requests_with_results_3d_plotly(json_path=None, data=None):
    """
    Highly interactive 3D scatter plot using Plotly
    """
    if data is None:
        with open(json_path, 'r') as f:
            data = json.load(f)
    
    coords = data["requests"]
    results = data["results"]
    
    # Extract coordinates
    x = [pt[0] for pt in coords]
    y = [pt[1] for pt in coords]
    z = [pt[2] for pt in coords]
    
    # Create color mapping - use numeric values for colorscale
    colors = [0 if not res else 1 for res in results]
    color_labels = ['Failed' if not res else 'Successful' for res in results]
    
    # Create Plotly 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            colorscale=[[0, 'red'], [1, 'blue']],
            opacity=0.8,
            line=dict(width=2, color='black'),
            showscale=True,
            colorbar=dict(
                title="Result",
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Failed", "Successful"]
            )
        ),
        text=[f'Point {i+1}<br>Result: {color_labels[i]}<br>Coords: ({x[i]:.2f}, {y[i]:.2f}, {z[i]:.2f})' 
              for i in range(len(results))],
        hoverinfo='text',
        name='Sembas Results'
    )])
    
    # Update layout
    fig.update_layout(
        title='Sembas Samples - Successful Races',
        scene=dict(
            xaxis_title='Progress Weight',
            yaxis_title='Bounds Weight',
            zaxis_title='Proximity Weight',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700
    )
    
    fig.show()
    return fig

# Function to load and plot from JSON file - FIXED
def load_and_plot_interactive(json_path):
    """
    Load data from JSON file and create both interactive plots
    """
    print(f"Loading data from {json_path}...")
    
    # Matplotlib version
    print("Creating matplotlib interactive plot...")
    fig_mpl, ax_mpl = plot_requests_with_results_3d_interactive(json_path)
    
    # Plotly version
    print("Creating Plotly interactive plot...")
    fig_plotly = plot_requests_with_results_3d_plotly(json_path)
    
    return fig_mpl, fig_plotly

# Example usage with file
if __name__ == "__main__":
    # Make sure to have your results.json file in the same directory
    fig_mpl, fig_plotly = load_and_plot_interactive('data/paper_data/scalar_collision_vector_pass.json')
    
    print("\nInteractive plots created!")
    print("- Matplotlib plot: Use mouse to rotate, zoom, and pan")
    print("- Plotly plot: Hover over points for details, use toolbar for controls")
    print("- To use with your own data: load_and_plot_interactive('your_file.json')")


# Sample data structure for reference
# sample_data_structure = {
#     "requests": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
#     "results": [True, False, True]
# }