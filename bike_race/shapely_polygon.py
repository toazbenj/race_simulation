import numpy as np
from shapely import LineString, Polygon
import matplotlib.pyplot as plt

def create_vehicle_polygon(
        x : float,
        y : float,
        length_rov : float,
        length_fov : float,
        width : float,
        phi : float,
) -> Polygon:
    """
    Creates a vehicle polygon based on geometric information

    :: Parameters ::
    x : float
        x position of the center of the vehicle.
    y : float
        y position of the venter of the vehicle
    length_rov : float
        Distance from the center of the vehicle to the rear of vehicle
    length_fov : float
        Distance from the center of the vehicle to the front of vehicle
    width : float
        Vehicle width.
    phi : float
        Vehicle heading (in degrees)
    """

    # Find the points of the rear and front of the vehicle
    rear = project_point(
        x,y,
        distance = length_rov,
        angle_radians = np.pi + np.deg2rad(phi)
    )
    front = project_point(
        x,y,
        distance = length_fov,
        angle_radians = np.deg2rad(phi)
    )

    # Create a line
    linestring = LineString([rear, front])
    
    # Buffer the line
    polygon = linestring2polygon(linestring, width)

    return polygon

def linestring2polygon(
        linestring : LineString, 
        width : float
    ) -> Polygon:
    """
    Transforms a @linestring into a Polygon given a @width
    by buffering the LineString with the given width to create a polygon
    """
    # return linestring.buffer(width / 2, cap_style=2, join_style=2)
    return linestring.buffer(width / 2, cap_style="flat", join_style="bevel")

def project_point(
        x : float, 
        y : float, 
        distance : float, 
        angle_radians : float
    ) -> list[float, float]:
    """
    Projects an @x,@y point @distance by @angle_radians

    Returns new x,y point.
    """
    new_x = x + distance * np.cos(angle_radians)
    new_y = y + distance * np.sin(angle_radians)
    return new_x, new_y

def main():
    p0 = (5,5)
    poly = create_vehicle_polygon(
        *p0,
        length_rov = 10,
        length_fov = 10,
        width = 10,
        phi = 45
    )
    
    p1 = (27,5)
    poly2 = create_vehicle_polygon(
        *p1,
        length_rov = 10,
        length_fov = 10,
        width = 10,
        phi = 0
    )

    print(
        "intersects?", 
        poly.intersects(poly2) 
    )

    plt.figure(figsize=(5,5))
    ax = plt.gca()

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal')

    ax.plot(
        *p0,
        marker = "+",
        color = "black"
    )

    x,y = poly.exterior.xy
    ax.fill(
        x,y, 
        facecolor="None",
        edgecolor = "blue"
    )

    ax.plot(
        *p1,
        marker = "+",
        color = "black"
    )

    x,y = poly2.exterior.xy
    ax.fill(
        x,y, 
        facecolor="None",
        edgecolor = "red"
    )

    plt.show()
    return

if __name__ == "__main__":
    main()