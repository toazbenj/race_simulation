import numpy as np
import time
import matplotlib.pyplot as plt
from cost_adjust_cvx import find_adjusted_costs  # Adjust module path as needed

def generate_smooth_cost_matrix(size, noise_scale=1.0):
    """
    Generate a matrix that varies smoothly along rows and columns.
    """
    base = np.outer(np.linspace(0, 1, size), np.linspace(0, 1, size)) * 10
    noise = np.random.normal(0, noise_scale, (size, size))
    return base + noise


def benchmark(start_size=3, end_size=10, step=1, min_successes=10, max_attempts=1000):
    sizes = list(range(start_size, end_size + 1, step))
    avg_times = []
    std_devs = []

    for size in sizes:
        success_times = []
        attempts = 0

        while len(success_times) < min_successes and attempts < max_attempts:
            A1 = generate_smooth_cost_matrix(size)
            B1 = generate_smooth_cost_matrix(size)
            C1 = generate_smooth_cost_matrix(size)
            D2 = generate_smooth_cost_matrix(size)

            start = time.time()
            E = find_adjusted_costs(A1, B1, C1, D2)
            end = time.time()

            attempts += 1
            if E is not None:
                success_times.append(end - start)

        if len(success_times) >= min_successes:
            avg_time = np.mean(success_times)
            std_dev = np.std(success_times)
            avg_times.append(avg_time)
            std_devs.append(std_dev)
            print(f"Size {size}x{size} | Avg: {avg_time:.4f}s | Std Dev: {std_dev:.4f}s | Successes: {len(success_times)}")
        else:
            avg_times.append(None)
            std_devs.append(None)
            print(f"Size {size}x{size} | Not enough valid runs (only {len(success_times)})")

    return sizes, avg_times, std_devs

def plot_results(sizes, times, stds):
    sizes_plot = [s for s, t in zip(sizes, times) if t is not None]
    times_plot = [t for t in times if t is not None]
    stds_plot = [s for s, t in zip(stds, times) if t is not None]

    plt.figure(figsize=(10, 6))
    plt.errorbar(sizes_plot, times_plot, yerr=stds_plot, fmt='o-', color='cyan',
                 ecolor='black', capsize=5, elinewidth=1.5, linewidth=2)
    plt.title('Runtime vs Matrix Size (with Error Bars)', fontsize=14)
    plt.xlabel('Matrix Size (N x N)', fontsize=12)
    plt.ylabel('Average Runtime (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(sizes_plot)
    plt.style.use('dark_background')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    sizes, avg_times, std_devs = benchmark(start_size=3, end_size=20, step=1, min_successes=10)
    plot_results(sizes, avg_times, std_devs)
