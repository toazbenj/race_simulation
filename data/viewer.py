import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

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
        st.markdown(f"## Vector Cost Method")

    with open(file, "r") as f:
        data = json.load(f)

    # Show all plots from this file in a row
    cols = st.columns(len(data))
    for i, col in enumerate(cols):
        with col:
            fig = plot_sembas_results(data, i)
            st.plotly_chart(fig, use_container_width=True)

if page_idx == 0:
    volumes = {
        "Cost Structure": ["Scalar", "Vector"],
        "Close Tail": [0.671, 0.866],
        "Far Tail": [0.510, 0.963],
        "Inside Edge": [0.472, 0.920],
        "Outside Edge": [0.682, 0.995],
    }
elif page_idx == 1:
    volumes = {
        "Cost Structure": ["Scalar", "Vector"],
        "Close Tail": [0.985, 1.000],
        "Far Tail": [0.935, 0.985],
        "Inside Edge": [0.961, 0.978],
        "Outside Edge": [1.000, 1.000],
    }
else:
    volumes = {
        "Cost Structure": ["Scalar", "Vector"],
        "Close Tail": [0.942, 1.000],
        "Far Tail": [0.934, 1.000],
        "Inside Edge": [0.923, 0.974],
        "Outside Edge": [0.909, 1.000],
    }

# Convert to DataFrame
df = pd.DataFrame(volumes)

# Set index
df.set_index("Cost Structure", inplace=True)

# Center alignment with Pandas Styler
styled_df = df.style.set_table_styles([
   {"selector": "th, td", "props": [("text-align", "center")]},
]).set_properties(**{"text-align": "center"}).format(precision=3)

# Streamlit app
st.title("Success Volumes")

st.dataframe(styled_df, width=500)
