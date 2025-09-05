import json
import streamlit as st
import plotly.graph_objects as go

def plot_sembas_results(data, num):
    """Single 3D scatter plot from a JSON dict entry"""
    coords = data[num]["requests"]
    results = data[num]["results"]
    description = data[num]["description"]

    x = [pt[0] for pt in coords]
    y = [pt[1] for pt in coords]
    z = [pt[2] for pt in coords]

    colors = [1 if res else 0 for res in results]

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=6,
            color=colors,
            colorscale=[[0, 'red'], [1, 'blue']],
            opacity=1.0,
            line=dict(width=1, color='black')
        ),
        text=[f'Point {i+1}<br>{"Successful" if res else "Failed"}<br>'
              f'({x[i]:.2f}, {y[i]:.2f}, {z[i]:.2f})'
              for i, res in enumerate(results)],
        hoverinfo='text'
    )])

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=6,
            color=colors,
            colorscale=[[0, 'red'], [1, 'blue']],
            opacity=1.0,
            line=dict(width=1, color='black')
        ),
        text=[f'Point {i+1}<br>{"Successful" if res else "Failed"}<br>'
              f'({x[i]:.2f}, {y[i]:.2f}, {z[i]:.2f})'
              for i, res in enumerate(results)],
        hoverinfo='text'
    )])

    formatted_title = description.split('>,')[-1].strip().replace('_', ' ')

    fig.update_layout(
        title=formatted_title,
        scene=dict(
            xaxis_title='Progress Weight',
            yaxis_title='Bounds Weight',
            zaxis_title='Proximity Weight'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=300,
    )

    return fig


# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Sembas Results Viewer")

# Hardcoded JSON file paths
files = [
    "data/visual_data/scalar_pass.json",
    "data/visual_data/vector_pass.json",
    "data/visual_data/scalar_bounds.json",
    "data/visual_data/vector_bounds.json",
    "data/visual_data/scalar_collision.json",
    "data/visual_data/vector_collision.json",
]

# Group files into pages (2 files per page)
pages = [
    files[0:2],
    files[2:4],
    files[4:6],
]

# Sidebar page selector
page = st.sidebar.radio("Select Page", ["Successful Passes", "Staying in Track Bounds", "Avoiding Collisions"])
# page_idx = int(page.split()[-1]) - 1
page_idx = ["Successful Passes", "Staying in Track Bounds", "Avoiding Collisions"].index(page)
selected_files = pages[page_idx]

# Show selected files
for i, file in enumerate(selected_files):
    
    if i % 2 == 0:
        st.markdown(f"## Scalarization Method")
    else:
        st.markdown(f"### Vector Cost Method")

    with open(file, "r") as f:
        data = json.load(f)

    # Show all plots from this file in a row
    cols = st.columns(len(data))
    for i, col in enumerate(cols):
        with col:
            fig = plot_sembas_results(data, i)
            st.plotly_chart(fig, use_container_width=True)