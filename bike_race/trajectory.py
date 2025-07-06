"""
Racing Simulation - Trajectory and Collision Detection Utilities

This module provides functions and classes for managing race trajectories, detecting collisions, and computing
cost-based trajectory decisions for simulated bicycles in a circular racecourse.

Key Features:
- **Bounding Box Computation:** Determines the minimum enclosing box around a trajectory for efficient collision checks.
- **Intersection Detection:** Checks if two bounding boxes or line segments intersect.
- **Trajectory Class:** Represents a sequence of positions forming a bicycle’s path and evaluates its cost based on
  distance, bounds, and collisions.
- **Cost-based Decision-Making:** Computes absolute and relative costs for selecting optimal trajectories.

Modules Used:
- math: Provides mathematical functions for angle and distance calculations.
- pygame: Handles rendering of trajectories and collisions.
- numpy: Enables numerical operations for cost calculations.
- constants: Stores simulation configuration values.

Functions:
- `bounding_box(points)`: Computes the bounding box of a set of points.
- `boxes_intersect(box1, box2)`: Checks if two bounding boxes intersect.
- `intersecting_area(box1, box2)`: Calculates the area of intersection between two bounding boxes.
- `intersect(line1, line2)`: Determines if two line segments intersect.

Class:
- `Trajectory`:
  - `__init__`: Initializes trajectory properties such as cost values and bounding boxes.
  - `draw(screen)`: Renders the trajectory and its associated cost.
  - `update()`: Updates trajectory costs and determines collision interactions.
  - `add_point(x, y)`: Adds a new point to the trajectory and updates cost metrics.
  - `get_bounding_box()`: Retrieves the bounding box of the trajectory.
  - `check_bounds(new_x, new_y)`: Checks if a point is within the racecourse boundaries.
  - `angel(x, y)`: Computes the angular position of a point on the racetrack.
  - `trajectory_intersection(other_traj)`: Determines if two trajectories intersect.
  - `absolute_trajectory_sensing(other_traj, action_interval, mpc_horizon)`: Detects absolute trajectory overlaps.
  - `relative_trajectory_sensing(other_traj)`: Computes relative costs between competing trajectories.

Entry Point:
- This module is designed to be imported and utilized within a larger racing simulation.
"""

from math import atan2, pi
import pygame
from constants import *

def bounding_box(points):
    """
    Compute the bounding box of a set of points.

    Parameters:
    - points (list[tuple[float, float]]): List of (x, y) coordinate points.

    Returns:
    - tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) defining the bounding box.
    """

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def boxes_intersect(box1, box2):
    """
    Check if two bounding boxes intersect.

    Parameters:
    - box1 (tuple[float, float, float, float]): (min_x, min_y, max_x, max_y) for box 1.
    - box2 (tuple[float, float, float, float]): (min_x, min_y, max_x, max_y) for box 2.

    Returns:
    - bool: True if the boxes intersect, False otherwise.
    """

    return not (box1[2] < box2[0] or box1[0] > box2[2] or
                box1[3] < box2[1] or box1[1] > box2[3])


def intersecting_area(box1, box2):
    """
    Calculate the area of intersection between two bounding boxes.

    Parameters:
    - box1 (tuple[float, float, float, float]): Bounding box 1.
    - box2 (tuple[float, float, float, float]): Bounding box 2.

    Returns:
    - float: The intersection area, or 0 if no intersection occurs.
    """

    # Calculate the intersection bounds
    inter_min_x = max(box1[0], box2[0])
    inter_min_y = max(box1[1], box2[1])
    inter_max_x = min(box1[2], box2[2])
    inter_max_y = min(box1[3], box2[3])

    # Compute the width and height of the intersection
    inter_width = max(0, inter_max_x - inter_min_x)
    inter_height = max(0, inter_max_y - inter_min_y)

    # Return the intersection area
    return inter_width * inter_height


def intersect(line1, line2):
    """
    Check if two line segments intersect.

    Parameters:
    - line1 (list[tuple[float, float]]): Two endpoints (x1, y1) and (x2, y2) of the first line segment.
    - line2 (list[tuple[float, float]]): Two endpoints (x3, y3) and (x4, y4) of the second line segment.

    Returns:
    - bool: True if the line segments intersect, False otherwise.
    """

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B = line1
    C, D = line2

    is_same_point = A == C or B == D or A == D or B == C
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D) or is_same_point

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def min_point_to_point_distance(line1, line2):
    min_dist = float('inf')
    for p1 in line1:
        for p2 in line2:
            dist = euclidean_distance(p1, p2)
            if dist < min_dist:
                min_dist = dist
    return min_dist


class Trajectory:
    def __init__(self,  course, bike, color, number=0):
        """
        Initializes a trajectory object that tracks a bicycle's movement.

        Parameters:
        - course (Course): The racetrack course.
        - bike (Bicycle): The bicycle associated with this trajectory.
        - color (tuple[int, int, int]): RGB color of the trajectory.
        - number (int, optional): Identifier for the trajectory. Default is 0.

        Returns:
        - None
        """
        self.min_x = 10000
        self.max_x = -10000
        self.min_y = 10000
        self.max_y = -10000

        self.color = color
        self.course = course
        self.bike = bike
        self.length = 0

        self.points = []
        self.intersecting_trajectories = []
        self.bounds_cost = 0

        self.relative_progress_costs =  np.zeros((len(self.bike.action_lst) ** self.bike.mpc_horizon))
        self.proximity_costs =  np.zeros((len(self.bike.action_lst) ** self.bike.mpc_horizon))

        self.is_displaying = False
        self.is_chosen = False
        self.is_collision_checked = False
        self.is_relative_checked = False
        self.number = number

        self.closest_boundary_distance = -1000
        self.closest_opponent_distance = -1000

        self.out_bounds_cnt = 0

    def draw(self, screen):
        """
        Draws the trajectory and its associated cost on the Pygame screen.

        Parameters:
        - screen (pygame.Surface): The Pygame surface to draw on.

        Returns:
        - None
        """
        for pt in self.points:
            pygame.draw.circle(screen, self.color, (pt[0], pt[1]), 1)

        # Draw costs near trajectories with spacing adjustment
        if self.is_displaying:
            font = pygame.font.Font(None, 20)
            # cost_text = font.render(f"{round(self.total_absolute_cost):.0f}", True, BLACK)
            num_text = font.render(f"{self.number}", True, BLACK)
            text_x = self.points[-1][0] + 10
            text_y = self.points[-1][1] - 10

            screen.blit(num_text, (text_x, text_y))

    def add_point(self, x, y):
        """
        Adds a new point to the trajectory and updates relevant cost metrics.

        Parameters:
        - x (float): X-coordinate of the new point.
        - y (float): Y-coordinate of the new point.

        Returns:
        - None
        """
        # update min/max of trajectory for bounding boxes
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)

        # self.bounds_cost += self.check_bounds(x, y)
        # boolean check if out of bounds for each point
        self.out_bounds_cnt |= self.check_bounds(x,y)
        new_bounds_cost = self.out_of_bounds_cost(x, y)
        if new_bounds_cost > self.bounds_cost:
            self.bounds_cost = new_bounds_cost

        if self.check_bounds(x,y) != 0:
            self.color = RED

        self.length = self.angle_displacement(self.bike.x, self.bike.y, x, y)
        self.points.append((round(x, 2), round(y, 2)))

    def out_of_bounds_cost(self, x, y):
        dx = x - WIDTH/2
        dy = y - HEIGHT/2
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # distance off track
        # if dist < INNER_RADIUS:
        #     # cost = abs(INNER_RADIUS - dist) ** 2
        #     cost = abs(INNER_RADIUS - dist)
        #
        # elif dist > OUTER_RADIUS:
        #     # cost = abs(dist - OUTER_RADIUS) ** 2
        #     cost = abs(dist - OUTER_RADIUS)
        # else:
        #     cost = 0
        # distance from center line

        # distance from race center line
        radius = (OUTER_RADIUS + INNER_RADIUS) / 2
        distance_to_perimeter = abs(dist - radius)

        cost = 1 - np.exp(-(2/BOUNDS_SPREAD * distance_to_perimeter) ** 2)

        return cost

    def get_bounding_box(self):
        """
        Retrieves the bounding box of the trajectory.

        Returns:
        - tuple[float, float, float, float]: Bounding box (min_x, min_y, max_x, max_y).
        """
        return self.min_x, self.min_y, self.max_x,  self.max_y

    def check_bounds(self, new_x, new_y):
        """
        Checks if a given point is within the racecourse boundaries.

        Parameters:
        - new_x (float): X-coordinate of the point.
        - new_y (float): Y-coordinate of the point.

        Returns:
        - int: 0 if within bounds, 1 if out of bounds.
        """
        # Calculate the distance from the center to the point
        distance_squared = (new_x - self.course.center_x) ** 2 + (new_y - self.course.center_y) ** 2

        # Check if the distance is between the outer radius and inner radius
        if INNER_RADIUS ** 2 <= distance_squared <= OUTER_RADIUS ** 2:
            return 0
        else:
            return 1

    def angle(self, x, y):
        """
        Computes the angular position of a point on the racetrack.

        Parameters:
        - x (float): X-coordinate of the point.
        - y (float): Y-coordinate of the point.

        Returns:
        - float: Angle in radians, normalized between [0, 2π).
        """
        # Calculate angular position in radians
        angle = atan2(y - self.course.center_y, x - self.course.center_x)
        angle = (angle + 2 * pi) % (2 * pi)  # Normalize to [0, 2π)
        return angle

    def angle_displacement(self, x1, y1, x2, y2):
        """
         Calculates the difference in angle covered by the trajectory.

         Parameters:
         - x (float): X-coordinate of the new position.
         - y (float): Y-coordinate of the new position.

         Returns:
         - float: Distance traveled along the track.
         """

        angle1 = self.angle(x1, y1)
        angle2 = self.angle(x2, y2)

        distance = abs(angle2 - angle1)
        return distance

    def trajectory_intersection(self, other_traj):
        """
        Checks if this trajectory intersects with another trajectory.

        Parameters:
        - other_traj (Trajectory): Another trajectory to check for intersections.

        Returns:
        - None
        """

        for (pt1, pt2) in zip(self.points[:-2], self.points[1:]):
            for (pt3, pt4) in zip(other_traj.points[:-2], other_traj.points[1:]):

                if intersect([pt1, pt2], [pt3, pt4]):
                    self.intersecting_trajectories.append(other_traj)

    def absolute_trajectory_sensing(self, other_traj, action_interval, mpc_horizon):
        """
        Detects whether this trajectory intersects with another trajectory using bounding box filtering.

        Parameters:
        - other_traj (Trajectory): The opponent's trajectory.
        - action_interval (int): The action execution interval.
        - mpc_horizon (int): Model predictive control horizon.

        Returns:
        - bool: True if the trajectories overlap, False otherwise.
        """

        # overlap
        # Compute bounding boxes
        box1 = self.get_bounding_box()
        box2 = other_traj.get_bounding_box()

        is_overlap = False
        # If bounding boxes don't overlap, trajectories don't intersect
        if boxes_intersect(box1, box2):
            # length must be multiple of action interval size
            length_interval = action_interval * mpc_horizon
            (pt1, pt2) = self.points[0], self.points[length_interval - 1]
            (pt3, pt4) = other_traj.points[0], other_traj.points[length_interval - 1]

            if intersect([pt1, pt2], [pt3, pt4]):
                self.intersecting_trajectories.append(other_traj)
                other_traj.intersecting_trajectories.append(self)
                self.color = ORANGE
                other_traj.color = ORANGE
                is_overlap = True

        return is_overlap

    def relative_trajectory_sensing(self, other_traj):
        """
        Computes relative cost values between competing trajectories.

        Parameters:
        - other_traj (Trajectory): The opponent's trajectory.

        Returns:
        - None
        """

        def unwrap_angle_difference(start_angle, end_angle):
            """Returns the signed minimal angular difference considering wraparound."""
            delta = end_angle - start_angle
            if delta < -pi:
                delta += 2 * pi
            elif delta > pi:
                delta -= 2 * pi
            return delta

        # Own trajectory angle delta
        self_start_angle = self.angle(*self.points[0])
        self_end_angle = self.angle(*self.points[-1])
        self_delta = unwrap_angle_difference(self_start_angle, self_end_angle)

        # Other trajectory angle delta
        other_start_angle = other_traj.angle(*other_traj.points[0])
        other_end_angle = other_traj.angle(*other_traj.points[-1])
        other_delta = unwrap_angle_difference(other_start_angle, other_end_angle)

        # Add prior state angle and completed laps
        self_total_angle = self_delta + self.bike.previous_angle + 2 * pi * self.bike.laps_completed
        other_total_angle = other_delta + other_traj.bike.previous_angle + 2 * pi * other_traj.bike.laps_completed

        # Relative arc-length difference
        angle_difference = other_total_angle - self_total_angle

        # Store cost bidirectionally
        self.relative_progress_costs[other_traj.number] = angle_difference
        other_traj.relative_progress_costs[self.number] = -angle_difference

    def proximity_sensing(self, other_traj):
        minimum_distance = min_point_to_point_distance(self.points, other_traj.points)

        cost = 0
        if minimum_distance < PROXIMITY_SPREAD:
            # cost = (threshold - minimum_distance) ** 2
            # cost = abs(threshold - minimum_distance)
            cost =  np.exp(-(2 * 1 / PROXIMITY_SPREAD * minimum_distance))

        self.proximity_costs[other_traj.number] = cost
        other_traj.proximity_costs[self.number] = cost