"""
Racing Course Simulation - Course Class

This module defines the `Course` class, which represents the race track and manages the simulation of two competing
bicycles. The track consists of an inner and outer boundary, forming a circular racecourse where bikes navigate
throughout the race.

Key Features:
- Initializes the race track with configurable radii and randomized start positions.
- Implements a centerline snapping algorithm to ensure bikes start at valid track positions.
- Manages two `Bicycle` objects that interact during the simulation.
- Updates the simulation state by processing bike movements, decisions, and interactions.
- Draws the race track and the bicycles using Pygame graphics.
- Implements a "Skip Race" button for user control.
- Saves race statistics, including passes, collisions, and performance metrics, in a CSV file.

Modules Used:
- pygame: For graphical rendering and user interaction.
- math: Provides mathematical functions for position calculations.
- random: Generates randomized race conditions.
- csv: Handles exporting race statistics to a CSV file.
- bicycle: Custom module representing individual racing bicycles.
- constants: External file containing configurable simulation parameters.

Class Methods:
- `__init__`: Initializes the course and places bikes at randomized or default positions.
- `draw_button`: Renders an interactive button on the screen.
- `snap_to_centerline`: Adjusts a point to align with the centerline of the track.
- `draw`: Renders the race track and bikes.
- `update`: Advances the simulation by processing bike movements.
- `save_stats`: Logs race performance data for later analysis.

"""
import math

import pygame
from math import atan2, pi, sqrt, ceil

from numpy.matrixlib.defmatrix import matrix

from bicycle import Bicycle
import random
import csv
from constants import *


def compute_angle(x, y, center_x, center_y):
    """
    Computes the bicycle's angle relative to the center of the track in a clockwise direction,
    starting from far right

    Returns:
    - float: Angle in radians normalized between [0, 2π].
    """
    dx = x - center_x
    dy = y - center_y

    # Compute angle normally (CCW) using atan2
    angle = atan2(dy, dx)

    # Normalize angle to the range [0, 2π]
    if angle < 0:
        angle += 2 * pi  # Ensures all angles stay positive in [0, 2π]

    return angle


class Course:
    def __init__(self, center_x, center_y, weights1, weights2, race_number, outer_radius=300, inner_radius=125,
                 randomize_start=False, seed=42):
        """
        Initializes the racecourse with specified track dimensions and randomization options.

        Parameters:
        - center_x (int): X-coordinate of the track center.
        - center_y (int): Y-coordinate of the track center.
        - outer_radius (int, optional): Outer boundary radius of the track.
        - inner_radius (int, optional): Inner boundary radius of the track.
        - randomize_start (bool, optional): Whether to randomize the initial positions of the bicycles.
        - seed (int, optional): Seed value for randomization. Default is 42.

        Returns:
        - None
        """
        self.race_number = race_number
        self.count = 0
        self.center_x = center_x
        self.center_y = center_y
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius

        # Calculate the track centerline radius
        self.centerline_radius = (inner_radius + outer_radius) / 2

        random.seed(seed)

        # Randomize bike start positions if enabled
        if randomize_start:
            # Generate random points in a wider area and snap them to the centerline
            while True:
                x1 = center_x + random.uniform(-outer_radius, outer_radius)
                y1 = center_y + random.uniform(-outer_radius, outer_radius)

                x2 = center_x + random.uniform(-outer_radius, outer_radius)
                y2 = center_y + random.uniform(-outer_radius, outer_radius)

                # Snap points to the centerline
                x1, y1 = self.snap_to_centerline(x1, y1)
                x2, y2 = self.snap_to_centerline(x2, y2)

                # make sure player 1 is behind player 2
                angle1 = compute_angle(x1, y1, self.center_x, self.center_y)
                angle2 = compute_angle(x2, y2, self.center_x, self.center_y)

                if angle1 < angle2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1

                # make sure they are close enough to interact, but not for spawn collision
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                max_distance = 2 * math.pi * self.centerline_radius/7
                if (distance < max_distance) and (distance > 2 * COLLISION_RADIUS) and (abs(angle1-angle2) < math.pi):
                    break
            
            # print(f'Spawn angles: {compute_angle(x1, y1, self.center_x, self.center_y)},\
            #                       {compute_angle(x2, y2, self.center_x, self.center_y)}')
            # print(f'Distance: {distance}')

        else:
            # Default start positions
            x1 = center_x
            y1 = center_y + self.centerline_radius

            x2 = center_x + 100
            y2 = center_y + self.centerline_radius

        phi1 = atan2(y1-center_y, x1-center_x) + pi/2
        phi2 = atan2(y2-center_y, x2-center_x) + pi/2

        # Initialize bikes facing directly down the track
        # Note switched x,y phi for lead/follow switch
        # self.bike1 = Bicycle(self, x=x2, y=y2, phi=phi2, is_relative_cost=True, velocity_limit=15)
        # self.bike2 = Bicycle(self, x=x1, y=y1, phi=phi1, color=GREEN,
        #                      is_vector_cost=False, is_relative_cost=True, velocity_limit=10, opponent=self.bike1)
        # self.bike1.opponent = self.bike2

        self.bike1 = Bicycle(self, x=x1, y=y1, phi=phi1, is_vector_cost=P1_IS_VECTOR_COST,
                             velocity_limit=DEFENDER_SPEED, theta_a=weights1[0], theta_b=weights1[1], theta_c=weights1[2])
        self.bike2 = Bicycle(self, x=x2, y=y2, phi=phi2, color=GREEN, is_vector_cost=P2_IS_VECTOR_COST, is_cost_populating=True,
                             velocity_limit=ATTACKER_SPEED, opponent=self.bike1,
                             theta_a=weights2[0], theta_b=weights2[1], theta_c=weights2[2])
        # bike must be initialized first before sharing information
        self.bike1.opponent = self.bike2

    def draw_button(self, screen, text, x, y, width, height, base_color, hover_color):
        """
        Draws a button on the Pygame screen and detects if it was clicked.

        Parameters:
        - screen (pygame.Surface): The Pygame screen to draw on.
        - text (str): Text to display on the button.
        - x (int): X-coordinate of the button's top-left corner.
        - y (int): Y-coordinate of the button's top-left corner.
        - width (int): Width of the button.
        - height (int): Height of the button.
        - base_color (tuple[int, int, int]): RGB color of the button in normal state.
        - hover_color (tuple[int, int, int]): RGB color of the button when cursor hovers.

        Returns:
        - bool: True if the button was clicked, False otherwise.
        """
        mouse_pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        # Change color on hover
        color = hover_color if x < mouse_pos[0] < x + width and y < mouse_pos[1] < y + height else base_color

        pygame.draw.rect(screen, color, (x, y, width, height), border_radius=10)

        # Render text
        font = pygame.font.Font(None, 36)
        text_surf = font.render(text, True, WHITE)
        text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
        screen.blit(text_surf, text_rect)

        # Check for click
        if click[0] == 1 and x < mouse_pos[0] < x + width and y < mouse_pos[1] < y + height:
            return True  # Button clicked

        return False

    def snap_to_centerline(self, x, y):
        """
        Adjusts a given point so it lies on the track’s centerline.

        Parameters:
        - x (float): X-coordinate of the original point.
        - y (float): Y-coordinate of the original point.

        Returns:
        - tuple[float, float]: Adjusted (x, y) coordinates snapped to the centerline.
        """
        dx = x - self.center_x
        dy = y - self.center_y
        distance = sqrt(dx ** 2 + dy ** 2)

        # Scale the point so it lands on the exact centerline radius
        if distance != 0:
            scale = self.centerline_radius / distance
            x = self.center_x + dx * scale
            y = self.center_y + dy * scale

        return x, y

    def draw(self, screen):
        """
        Draws the race track and the bicycles onto the Pygame screen.

        Parameters:
        - screen (pygame.Surface): The Pygame screen to draw on.

        Returns:
        - None
        """
        screen.fill(DARK_GREEN)
        pygame.draw.circle(screen, GRAY, (self.center_x, self.center_y), self.outer_radius)
        pygame.draw.circle(screen, DARK_GREEN, (self.center_x, self.center_y), self.inner_radius)
        pygame.draw.circle(screen, BLACK, (self.center_x, self.center_y), self.outer_radius, 3)
        pygame.draw.circle(screen, BLACK, (self.center_x, self.center_y), self.inner_radius, 3)

        self.bike1.draw(screen)
        self.bike2.draw(screen)

    def update(self):
        """
        Updates the simulation state by processing bike movements and decision-making. Choice creation and selection are
        separate since the second bike may make choices based on the first one's expected pick.

        Parameters:
        - None

        Returns:
        - None
        """

        self.bike1.update_choices(self.count, self.bike2)
        self.bike2.update_choices(self.count, self.bike1)

        self.bike1.update_action(self.count)
        self.bike2.update_action(self.count)

        if self.count % (ACTION_INTERVAL* MPC_HORIZON) == 0:
            self.save_costs(self.count / (ACTION_INTERVAL* MPC_HORIZON))

        self.count += 1

    def save_stats(self, seed):
        """
        Saves race statistics, such as passes, collisions, and performance metrics, to a CSV file.

        Parameters:
        - race_number (int): The index of the race being recorded.
        - seed (int): The seed used for race randomization.

        Returns:
        - None
        """

        with open(RACE_DATA, mode='a', newline='') as file:
            writer = csv.writer(file)
            if self.race_number == 0:
                writer.writerow(["Race Number", 'Passes P1', 'Passes P2', 'Collisions',
                                 'Choices',
                                 'Proportion Ahead P1', 'Proportion Ahead P2',
                                 'Win P1', 'Win P2',
                                 'Progress P1', 'Progress P2',
                                 'Out of Bounds P1', 'Out of Bounds P2',
                                 'Progress Cost P1', 'Progress Cost P2',
                                 'Bounds Cost P1', 'Bounds Cost P2',
                                 'Adjustment Count P2', f'Seed: {seed}'])

            try:
                p1_ahead = round(self.bike1.ahead_cnt/self.bike1.choice_cnt,2)
                p2_ahead = round(self.bike2.ahead_cnt/self.bike1.choice_cnt,2)
            except ZeroDivisionError:
                p1_ahead = 0
                p2_ahead = 0

            # how many times a single collision is registered (how hard the hit was)
            collision_amount = 200
            # P1 always ahead, pass count -1 since counts as a pass
            writer.writerow([self.race_number+1, self.bike1.pass_cnt, self.bike2.pass_cnt,
                             ceil(self.bike1.collision_cnt/collision_amount), self.bike1.choice_cnt,
                             p1_ahead, p2_ahead,
                             self.bike1.is_ahead, self.bike2.is_ahead,
                             round(self.bike1.progress_cnt), round(self.bike2.progress_cnt),
                             self.bike1.out_bounds_cnt, self.bike2.out_bounds_cnt,
                             round(self.bike1.progress_cost), round(self.bike2.progress_cost),
                             round(self.bike1.bounds_cost, 2), round(self.bike2.bounds_cost,2),
                             self.bike2.adjust_cnt])


    def save_costs(self, decision_number):
        """
        Save scalar and full 2D matrix features into CSV, expanding headers for both dimensions.
        """

        with open(COST_DATA, mode='a', newline='') as file:
            writer = csv.writer(file)

            if decision_number==0 and self.race_number==0:
                # Scalar features header
                header = [
                    'Theta_a1', 'Theta_b1', 'Theta_c1',
                    'Theta_a2', 'Theta_b2', 'Theta_c2',
                    'action1', 'Action_space1',
                    'action2', 'Action_space2'
                ]

                # For bike1 matrices: A, B, C (assuming they are 2D arrays)
                header += self.matrix_print(self.bike1.A, 'A1')
                header += self.matrix_print(self.bike1.B, 'B1')
                header += self.matrix_print(self.bike1.C, 'C1')

                # For bike2 matrices: A, B, C
                header += self.matrix_print(self.bike2.A, 'A2')
                header += self.matrix_print(self.bike2.B, 'B2')
                header += self.matrix_print(self.bike2.C, 'C2')


                # For state vectors (assuming these are 2D arrays; if 1D, adjust accordingly)
                state_dict = {1:'x', 2:'y', 3:'v', 4:'phi', 5:'b'}
                header += [f'State1_{state_dict[i]}' for i in range(1,self.bike1.state.shape[0]+1)]
                header += [f'State2_{state_dict[i]}' for i in range(1,self.bike2.state.shape[0]+1)]

                writer.writerow(header)

            # --- Build row data ---
            row = [
                self.bike1.theta_a, self.bike1.theta_b, self.bike1.theta_c,
                self.bike2.theta_a, self.bike2.theta_b, self.bike2.theta_c,
                self.bike1.action_index, self.bike1.action_space_size,
                self.bike2.action_index, self.bike2.action_space_size,
            ]

            # Append full matrix data by flattening in row-major order:
            row += self.bike1.A.flatten().tolist()
            row += self.bike1.B.flatten().tolist()
            row += self.bike1.C.flatten().tolist()
            row += self.bike2.A.flatten().tolist()
            row += self.bike2.B.flatten().tolist()
            row += self.bike2.C.flatten().tolist()
            row += self.bike1.state.flatten().tolist()
            row += self.bike2.state.flatten().tolist()

            writer.writerow(row)

    def matrix_print(self, matrix, name):
        line = [f'{name}_{i}_{j}' for i in range(1,matrix.shape[0]+1) for j in range(1,matrix.shape[1]+1)]
        return line