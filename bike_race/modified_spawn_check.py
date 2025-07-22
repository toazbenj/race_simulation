import matplotlib.pyplot as plt
import numpy as np

def plot_circle_points_by_angle(radius, start_point, angle_increment_rad, center=(0, 0), inner_radius_ratio=1.115):
    """
    Plots points on a circle and a second inner circle using the same angles.

    Parameters:
    - radius (float): Outer circle radius.
    - start_point (tuple): Starting (x, y) point on the outer circle.
    - angle_increment_rad (float): Angular step between points (in radians).
    - center (tuple): Center of the circle.
    - inner_radius_ratio (float): Ratio of inner radius to outer radius.
    """
    h, k = center
    x0, y0 = start_point

    # Compute starting angle from center to start_point
    start_angle = np.arctan2(y0 - k, x0 - h)

    # Compute number of points
    total_angle = 2 * np.pi
    num_points = int(np.ceil(total_angle / angle_increment_rad))

    # Generate angles
    angles = [start_angle + i * angle_increment_rad for i in range(num_points)]

    # Generate outer and inner circle points
    outer_points = [(h + radius * np.cos(angle), k + radius * np.sin(angle)) for angle in angles]
    inner_radius = radius * inner_radius_ratio
    inner_points = [(h + inner_radius * np.cos(angle), k + inner_radius * np.sin(angle)) for angle in angles]

    # Plot circles
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle(center, radius, color='lightgray', fill=False, linestyle='--'))
    ax.add_patch(plt.Circle(center, inner_radius, color='lightblue', fill=False, linestyle='--'))

    # Plot outer and inner points
    outer_xs, outer_ys = zip(*outer_points)
    inner_xs, inner_ys = zip(*inner_points)
    ax.plot(outer_xs, outer_ys, 'ro', label='Outer Points')
    ax.plot(inner_xs, inner_ys, 'go', label='Inner Points (Smaller Radius)')

    # Mark the start point
    ax.plot([x0], [y0], 'bo', label='Start Point')

    ax.set_aspect('equal')
    ax.set_title(f"Points at Two Radii (Δθ = {np.degrees(angle_increment_rad):.1f}°)")
    ax.legend()
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Print first few inner points (translated coordinates)
    print("Translated coordinates of inner points:")
    for pt in inner_points[:7]:
        x, y = pt
        print((x + 700, -y + 425))

# Example usage
radius = 325
start_point = (0, -325)  # start angle at 0 radians
angle_increment_rad = 0.123
plot_circle_points_by_angle(radius, start_point, angle_increment_rad)
