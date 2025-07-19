import matplotlib.pyplot as plt
import numpy as np

def plot_circle_points_by_angle(radius, start_point, angle_increment_rad, center=(0, 0)):
    """
    Plots points on a circle with a specified angular increment, starting from a given point.

    Parameters:
    - radius (float): Radius of the circle.
    - start_point (tuple): Starting (x, y) point on the circle.
    - angle_increment_rad (float): Angular step between points (in radians).
    - center (tuple): Center of the circle, default is (0, 0).
    """
    h, k = center
    x0, y0 = start_point

    # Compute the starting angle from the center
    start_angle = np.arctan2(y0 - k, x0 - h)

    # Calculate how many steps it takes to go ~360 degrees
    total_angle = 2 * np.pi
    num_points = int(np.ceil(total_angle / angle_increment_rad))

    # Generate points
    points = []
    for i in range(num_points):
        angle = start_angle + i * angle_increment_rad
        x = h + radius * np.cos(angle)
        y = k + radius * np.sin(angle)
        points.append((x, y))

    # Plot the circle
    circle = plt.Circle(center, radius, color='lightgray', fill=False, linestyle='--')
    fig, ax = plt.subplots()
    ax.add_patch(circle)

    # Plot the points
    xs, ys = zip(*points)
    ax.plot(xs, ys, 'ro')  # red dots

    # Mark the start point
    ax.plot([x0], [y0], 'bo', label='Start Point')  # blue dot

    ax.set_aspect('equal')
    ax.set_title(f"Points on Circle (Δθ = {np.degrees(angle_increment_rad):.1f}°)")
    ax.legend()
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    for pt in points[:7]:
        x = pt[0]
        y = pt[1]
        
        print((x+700, -y+425))

# Example usage
radius = 325
start_point = (0, -325)  # on the circle at 0 radians
# angle_increment_deg = 30
# angle_increment_rad = np.radians(angle_increment_deg)
angle_increment_rad = 0.123


plot_circle_points_by_angle(radius, start_point, angle_increment_rad)
