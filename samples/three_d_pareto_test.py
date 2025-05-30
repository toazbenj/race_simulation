import numpy as np
import plotly.graph_objects as go


def pareto_optimal(A1, B1, C1, column):
    """
    Identifies Pareto-optimal rows in two cost matrices where no other row is strictly better.

    Parameters:
    - A1 (np.ndarray): Cost matrix for player 1.
    - B1 (np.ndarray): Cost matrix for player 2.
    - column (int): The column index for evaluating Pareto optimality.

    Returns:
    - np.ndarray: Indices of Pareto-optimal rows.
    """

    num_rows = A1.shape[0]
    pareto_indices = []

    for i in range(num_rows):
        dominated = False
        for j in range(num_rows):
            if i != j:
                # Check if row j dominates row i

                # if (A1[j][column] <= A1[i][column] and B1[j][column] <= B1[i][column]) and (
                #         A1[j][column] < A1[i][column] or B1[j][column] < B1[i][column]):
                #     dominated = True
                #     break

                dominates = (
                        A1[j][column] <= A1[i][column] and
                        B1[j][column] <= B1[i][column] and
                        C1[j][column] <= C1[i][column]
                )
                strictly_better = (
                        A1[j][column] < A1[i][column] or
                        B1[j][column] < B1[i][column] or
                        C1[j][column] < C1[i][column]
                )
                print(f"dominates: {dominated}, strictly better: {strictly_better}")
                if dominates and strictly_better:
                    dominated = True
                    break
        if not dominated:
            pareto_indices.append(i)

    return np.array(pareto_indices)


# main

# Sample 3x3 cost matrices
A1 = np.array([[5, 2, 1],
               [5, 3, 2],
               [5, 1, 3]])

B1 = np.array([[3, 3, 2],
               [2, 1, 3],
               [3, 2, 1]])

C1 = np.array([[2, 1, 3],
               [3, 2, 1],
               [1, 3, 2]])

# Apply on column 0
pareto_idx = pareto_optimal(A1, B1, C1, column=0)
print(pareto_idx)

# Extract coordinates
x = A1[:, 0]
y = B1[:, 0]
z = C1[:, 0]

# Pareto optimal points
x_opt = x[pareto_idx]
y_opt = y[pareto_idx]
z_opt = z[pareto_idx]

# Plotly 3D scatter plot
fig = go.Figure()

# All points
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                           marker=dict(size=6, color='lightgray'),
                           name='All Points'))

# Pareto optimal points
fig.add_trace(go.Scatter3d(x=x_opt, y=y_opt, z=z_opt, mode='markers',
                           marker=dict(size=8, color='red', symbol='diamond'),
                           name='Pareto Optimal'))

fig.update_layout(
    scene=dict(
        xaxis_title='Cost A',
        yaxis_title='Cost B',
        zaxis_title='Cost C'
    ),
    title='Pareto Optimality in 3D Cost Space'
)

fig.show()
print()