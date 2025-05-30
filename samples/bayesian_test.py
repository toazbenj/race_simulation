import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from scipy.spatial.distance import cdist
from mpl_toolkits.axes_grid1 import make_axes_locatable

# LQR Cost Function
def lqr_cost(A, B, K, Q, R, z0, steps=20):
    cost = 0.0
    A_cl = A - B @ K
    z = z0
    for _ in range(steps):
        cost += (z.T @ (Q + K.T @ R @ K) @ z).item()
        z = A_cl @ z
    return cost


def eval_log_costs(A, B, points):
    return np.array([np.log(lqr_cost(A, B, k.reshape(1, -1), Q, R, z0)) for k in points])


# System Definitions
A_true = np.array([[0.785, -0.260], [-0.260, 0.315]])
B_true = np.array([[1.475], [0.607]])
A_hat = np.array([[0.700, -0.306], [-0.306, 0.342]])
B_hat = np.array([[1.543], [0.524]])

Q = np.eye(2)
R = np.array([[1]])
z0 = np.array([[1.0], [1.0]])

# Grid Setup
x1_range = np.linspace(-0.5, 4.5, 26)
x2_range = np.linspace(-3.5, 1.5, 26)
X1, X2 = np.meshgrid(x1_range, x2_range)
grid_points = np.c_[X1.ravel(), X2.ravel()]

# Initial Safe Points
fL_values = eval_log_costs(A_hat, B_hat, grid_points)
safe_indices = np.argsort(fL_values)[:3]
safe_points = grid_points[safe_indices]
f_values_safe = eval_log_costs(A_true, B_true, safe_points)

# SAFESLOPE Parameters
max_iters = 10
h = 0.75
beta = 3.0
true_opt_value = np.min(eval_log_costs(A_true, B_true, grid_points))

# kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-4)
kernel = C(1.0, (1e-2, 1e2)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e-1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

# SAFESLOPE Loop
safe_set = safe_points.copy()
safe_values = f_values_safe.copy()
cumulative_regret = []
cumulative_unsafe = []
searched_points = []
unsafe_points = []

for t in range(max_iters):
    gp.fit(safe_set, safe_values)
    mu, sigma = gp.predict(grid_points, return_std=True)
    lower = mu - np.sqrt(beta) * sigma
    upper = mu + np.sqrt(beta) * sigma

    # Safe expansion using local slope estimates
    expanded_safe_mask = np.full(len(grid_points), False)
    for i, x in enumerate(grid_points):
        for xj in safe_set:
            direction = x - xj
            norm_dir = np.linalg.norm(direction)
            if norm_dir == 0:
                continue
            unit_dir = direction / norm_dir

            # Finite difference estimate of slope along this direction
            x_proj = xj + 0.1 * unit_dir
            if np.any(x_proj < [x1_range[0], x2_range[0]]) or np.any(x_proj > [x1_range[-1], x2_range[-1]]):
                continue
            f_proj = np.log(lqr_cost(A_true, B_true, x_proj.reshape(1, -1), Q, R, z0))
            slope_est = (f_proj - eval_log_costs(A_true, B_true, [xj])[0]) / 0.1

            # Use slope to estimate upper bound at x
            f_xj_upper = upper[np.argmin(np.linalg.norm(grid_points - xj, axis=1))]
            predicted_upper = f_xj_upper + slope_est * norm_dir

            # print(predicted_upper)
            if predicted_upper <= h:
                expanded_safe_mask[i] = True
                break

    safe_candidates = grid_points[expanded_safe_mask]

    if len(safe_candidates) == 0:
        break

    width = upper[expanded_safe_mask] - lower[expanded_safe_mask]
    idx = np.argmax(width + 1e-6 * np.random.randn(*width.shape))
    x_next = safe_candidates[idx].reshape(1, -1)

    true_val = np.log(lqr_cost(A_true, B_true, x_next.reshape(1, -1), Q, R, z0))
    searched_points.append(x_next.flatten().copy())

    if true_val <= h:
        safe_set = np.vstack([safe_set, x_next])
        safe_values = np.append(safe_values, true_val)
        cumulative_unsafe.append(cumulative_unsafe[-1] if cumulative_unsafe else 0)
    else:
        cumulative_unsafe.append((cumulative_unsafe[-1] if cumulative_unsafe else 0) + 1)
        unsafe_points.append(x_next.flatten().copy())

    regret = true_val - true_opt_value
    cumulative_regret.append((cumulative_regret[-1] if cumulative_regret else 0) + regret)

    print(f"Evaluated K: {x_next.flatten()}, Log Cost = {true_val:.4f}, Safe = {true_val <= h}")
    print(f"Iteration {t}: Safe candidates = {len(safe_candidates)}")
    print(f"Iteration {t}: Safe candidates = {np.sum(expanded_safe_mask)}")
    print(f"Safe set size: {len(safe_set)}")

searched_points = np.array(searched_points)
unsafe_points = np.array(unsafe_points)


# Plotting
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
axs[0].plot(cumulative_regret)
axs[0].set_title("SAFESLOPE: Cumulative Regret")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Cumulative Regret")

axs[1].plot(cumulative_unsafe)
axs[1].set_title("SAFESLOPE: Cumulative Unsafe Samples")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Count")

cs = axs[2].contourf(X1, X2, fL_values.reshape(X1.shape), levels=30)
axs[2].scatter(searched_points[:, 0], searched_points[:, 1], color='yellow', label='Searched')
axs[2].scatter(safe_set[:, 0], safe_set[:, 1], color='green', label='Safe')
if len(unsafe_points) > 0:
    axs[2].scatter(unsafe_points[:, 0], unsafe_points[:, 1], color='red', label='Unsafe')

axs[2].set_title('Low-Fidelity Log-Cost over Controller Gain Grid')
axs[2].set_xlabel('K1')
axs[2].set_ylabel('K2')
axs[2].legend()

divider = make_axes_locatable(axs[2])
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(cs, cax=cax)

plt.tight_layout()
plt.show()

print()
