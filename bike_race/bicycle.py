"""
Bicycle Racing Simulation - Bicycle Class

This module defines the `Bicycle` class, which simulates a racing bicycle navigating a circular track. The bicycle
follows a simplified kinematic bicycle model and interacts with an opponent during the race.

Key Features:
- Simulates realistic bicycle motion using acceleration, steering, and velocity constraints.
- Implements trajectory planning with multiple possible action sequences.
- Uses cost-based decision-making to select optimal paths.
- Tracks race progress, collisions, and performance metrics.
- Detects lap completions and updates racing statistics.

Modules Used:
- math: Provides mathematical functions for kinematics.
- pygame: Handles graphical rendering of bicycles and trajectories.
- numpy: Enables matrix operations for cost calculations.
- constants: Stores simulation configuration parameters.
- cost_adjust_cvx: Computes cost adjustments for trajectory optimization.
- trajectory: Manages trajectory generation and visualization.
- itertools: Generates combinations of possible action sequences.

Class Methods:
- `__init__`: Initializes the bicycle with position, velocity, and opponent interactions.
- `compute_angle`: Determines the bicycle’s angle relative to the track’s center.
- `dynamics`: Updates the bicycle’s state based on acceleration and steering inputs.
- `draw`: Renders the bicycle, its past trajectory, and predicted future paths.
- `update_choices`: Refreshes available action choices at regular intervals.
- `update_action`: Executes the chosen action sequence and updates movement.
- `build_arr`: Constructs a cost array for trajectory evaluation.
- `build_vector_arr`: Computes vector-based cost matrices for decision-making.
- `compute_action`: Determines the optimal action sequence using cost analysis.
- `update_stats`: Tracks race statistics, including lap progress and collisions.
- `new_choices`: Generates new trajectory options based on possible actions.
- `check_lap_completion`: Detects lap completions when the bicycle crosses the finish line.

Entry Point:
- This module is designed to be integrated with a larger simulation and does not run independently.

"""

from math import cos, sin, tan, atan2, sqrt, pi
import pygame
from constants import *
from cost_adjust_cvx import find_adjusted_costs
from trajectory import Trajectory
from itertools import product

def generate_combinations(numbers, num_picks):
    """
    Generate all possible combinations of choices by picking `num_picks` times from the list `numbers`.

    Parameters:
    - numbers (list[int]): The list of numbers to pick from.
    - num_picks (int): The number of times to pick.

    Returns:
    - list[list[int]]: All possible combinations of length `num_picks`.
    """
    if not numbers or num_picks <= 0:
        return []

    # Use itertools.product to generate combinations
    combinations = list(product(numbers, repeat=num_picks))
    combinations = [list(element) for element in combinations]
    return combinations


class Bicycle:
    def __init__(self, course, x=300, y=300, v=5, color=BLUE, phi=radians(90), b=0, velocity_limit=15,
                 is_vector_cost=False, is_cost_populating=False, opponent=None, theta_a=1, theta_b=1, theta_c=1):
        """
            Initializes a Bicycle object to simulate movement on a racetrack.

            Parameters:
            - course (Course): The racetrack the bicycle is on.
            - x (float, optional): Initial x-coordinate. Default is 300.
            - y (float, optional): Initial y-coordinate. Default is 300.
            - v (float, optional): Initial velocity. Default is 5.
            - color (tuple[int, int, int], optional): RGB color of the bicycle. Default is BLUE.
            - phi (float, optional): Initial heading angle in radians. Default is 90 degrees.
            - b (float, optional): Initial slip angle. Default is 0.
            - velocity_limit (float, optional): Maximum velocity. Default is 15.
            - is_vector_cost (bool, optional): Whether to use vector cost calculation. Default is False.
            - is_relative_cost (bool, optional): Whether to use relative cost calculation. Default is False.
            - opponent (Bicycle, optional): Reference to the opponent bicycle. Default is None.

            Returns:
            - None
        """
        self.bicycle_size = BIKE_SIZE
        self.color = color

        self.x = x
        self.y = y
        self.v = v
        self.phi = phi
        self.b = b

        self.lr = LR
        self.lf = LF

        self.a = 0
        self.steering_angle = 0
        self.velocity_limit = velocity_limit

        self.past_trajectories = []  # Store past positions
        self.choice_trajectories = [] # upcoming possible traj
        self.action_choices = [] # sequences of actions to create all possible trajectories
        self.chosen_action_sequence = [] # sequence of actions to create chosen trajectory

        self.action_interval = ACTION_INTERVAL
        self.mpc_horizon = MPC_HORIZON
        self.action_lst = ACTION_LST

        self.course = course

        # Cost data
        self.action_space_size = len(ACTION_LST) ** MPC_HORIZON
        self.A =  np.zeros((self.action_space_size,self.action_space_size))
        self.B =  np.zeros((self.action_space_size,self.action_space_size))
        self.C =  np.zeros((self.action_space_size,self.action_space_size))
        self.action_index = 0
        self.state = np.zeros(5)
        self.theta_a = theta_a
        self.theta_b = theta_b
        self.theta_c = theta_c

        self.new_choices()
        self.is_vector_cost = is_vector_cost
        self.is_cost_populating = is_cost_populating
        self.opponent = opponent
        self.composite_cost_arr = None

        # stats
        self.pass_cnt = 0
        self.collision_cnt = 0
        self.choice_cnt = 0
        self.ahead_cnt = 0
        self.is_ahead = True
        self.progress_cnt = 0
        self.out_bounds_cnt = 0
        self.adjust_cnt = 0
        self.collision_radius = COLLISION_RADIUS

        self.progress_cost = 0
        self.bounds_cost = 0
        self.proximity_cost = 0

        # Lap tracking
        self.laps_completed = 0
        self.previous_angle = self.compute_angle()  # Initial angle
        self.is_crossing_finish = False

    def compute_angle(self):
        """
        Computes the bicycle's angle relative to the center of the track in a clockwise direction,
        starting from far right

        Returns:
        - float: Angle in radians normalized between [0, 2π].
        """
        dx = self.x - self.course.center_x
        dy = self.y - self.course.center_y

        # Compute angle normally (CCW) using atan2
        angle = atan2(dy, dx)

        # Normalize angle to the range [0, 2π]
        if angle < 0:
            angle += 2 * pi  # Ensures all angles stay positive in [0, 2π]

        return angle

    def dynamics(self, acc, steering, x_in, y_in, v_in, phi_in, b_in):
        """
        Computes the next state of the bicycle using a simple bicycle model.

        Parameters:
        - acc (float): Acceleration.
        - steering (float): Steering angle.
        - x_in (float): Current x-coordinate.
        - y_in (float): Current y-coordinate.
        - v_in (float): Current velocity.
        - phi_in (float): Current heading angle.
        - b_in (float): Current slip angle.

        Returns:
        - tuple[float, float, float, float, float]: Updated (x, y, v, phi, b).
        """
        # Update positions
        x_next = x_in + v_in * cos(phi_in + b_in) * DT
        y_next = y_in + v_in * sin(phi_in + b_in) * DT

        # Update heading angle
        phi_next = phi_in + (v_in / self.lr) * sin(b_in) * DT

        # Update velocity
        v_next = v_in + acc * DT
        # velocity limit
        if v_next > self.velocity_limit:
            v_next = self.velocity_limit
        v_next = max(0, v_next)  # Prevent negative velocity

        b_next = atan2(self.lr * tan(steering), self.lr + self.lf)

        self.state = np.round(np.array([x_next, y_next, v_next, phi_next, b_next]), decimals=6)
        return x_next, y_next, v_next, phi_next, b_next


    def draw(self, screen):
        """
        Draws the bicycle and its past and possible future trajectories.

        Parameters:
        - screen (pygame.Surface): The Pygame surface to draw on.

        Returns:
        - None
        """
        # Draw the bike
        points = [
            (self.x + self.bicycle_size * cos(self.phi) - self.bicycle_size / 2 * sin(self.phi),
             self.y + self.bicycle_size * sin(self.phi) + self.bicycle_size / 2 * cos(self.phi)),
            (self.x - self.bicycle_size * cos(self.phi) - self.bicycle_size / 2 * sin(self.phi),
             self.y - self.bicycle_size * sin(self.phi) + self.bicycle_size / 2 * cos(self.phi)),
            (self.x - self.bicycle_size * cos(self.phi) + self.bicycle_size / 2 * sin(self.phi),
             self.y - self.bicycle_size * sin(self.phi) - self.bicycle_size / 2 * cos(self.phi)),
            (self.x + self.bicycle_size * cos(self.phi) + self.bicycle_size / 2 * sin(self.phi),
             self.y + self.bicycle_size * sin(self.phi) - self.bicycle_size / 2 * cos(self.phi))
        ]
        pygame.draw.polygon(screen, self.color, points)

        # Draw the past trajectory
        if len(self.past_trajectories) > 1:
            for traj in self.past_trajectories:
                traj.draw(screen)

        for i, traj in enumerate(set(self.choice_trajectories)):
            # print(f"Trajectory {i}: Cost = {traj.cost}, Points = {traj.points[0]}, {traj.points[-1]}")
            traj.draw(screen)

        # check where opponent is, determine collision in every frame
        x2, y2 = self.opponent.x, self.opponent.y
        distance = sqrt((x2 - self.x) ** 2 + (y2 - self.y) ** 2)
        if distance < self.collision_radius:
            self.collision_cnt += 1


    def update_choices(self, count, other_bike):
        """
        Updates the available choices for the bicycle at regular intervals.

        Parameters:
        - count (int): Current simulation time step.
        - other_bike (Bicycle): The opponent bicycle.

        Returns:
        - None
        """
        if count % (self.action_interval * self.mpc_horizon) == 0:
            self.new_choices(other_bike)

    def update_action(self, count):
        """
        Updates the bicycle's action and moves it forward in time.

        Parameters:
        - count (int): Current simulation time step.

        Returns:
        - None
        """
        # Periodically compute actions
        if count % (self.action_interval * self.mpc_horizon) == 0:
            # self.new_choices()
            self.compute_action()
        # switch actions after action interval elapses
        if count % self.action_interval == 0:
            self.a = self.chosen_action_sequence[0][0] * ACCELERATION_INCREMENT
            self.steering_angle = self.chosen_action_sequence[0][1] * STEERING_INCREMENT
            self.chosen_action_sequence.remove(self.chosen_action_sequence[0])

        # Update the bicycle state
        self.x, self.y, self.v, self.phi, self.b  = self.dynamics(self.a, self.steering_angle, self.x, self.y, self.v, self.phi, self.b)

        # Check if a lap has been completed
        self.check_lap_completion()

    def build_arr(self, trajectories):
        """
        Builds a cost array for trajectory evaluation. Costs are calculated separately and combined in weighted sums.

        Parameters:
        - trajectories (list[Trajectory]): List of possible future trajectories.

        Returns:
        - None
        """
        self.C = np.zeros_like(self.A)  # reset C
        for i, traj in enumerate(trajectories):
            # indicator functions not scaled by weight
            self.A[i] = np.round(traj.relative_progress_costs, decimals=5)
            self.B[i] = np.full(self.B.shape[0], traj.bounds_cost)

            # decimal proximity cost
            self.C[i] = np.round(traj.proximity_costs, decimals=5)

            # # binary collision cost
            # for other_traj in traj.intersecting_trajectories:
            #     # print(other_traj.number)
            #     self.C[i][other_traj.number] += 1

        # print(self.color)
        # print(self.C)

        if self.is_vector_cost:
            E = find_adjusted_costs(self.A, self.B, self.C, self.opponent.composite_cost_arr)

            if E is None:
                # print("no minima")
                self.composite_cost_arr = self.A * self.theta_a + self.B * self.theta_b + self.C * self.theta_c
            else:
                print("adjustment success")
                self.composite_cost_arr = self.A + E
                self.adjust_cnt += 1
        else:
            self.composite_cost_arr = self.A * self.theta_a + self.B * self.theta_b + self.C * self.theta_c

    def compute_action(self):
        """
        Computes the best action sequence to take based on cost calculations. Uses security policies.

        Returns:
        - None
        """

        self.build_arr(self.choice_trajectories)
        self.action_index = np.argmin(np.max(self.composite_cost_arr, axis=1))
        print("Action: " + str(self.action_index))

        chosen_traj = self.choice_trajectories[self.action_index]
        chosen_traj.color = self.color
        chosen_traj.is_displaying = False
        chosen_traj.is_chosen = True

        self.past_trajectories.append(chosen_traj)
        self.choice_trajectories.remove(chosen_traj)
        self.chosen_action_sequence = self.action_choices[self.action_index]

        self.update_stats()

    def update_stats(self):
        """
        Updates statistical metrics of the bicycle such as lap progress, collisions, and passes.

        Returns:
        - None
        """

        # update stats
        self.choice_cnt += 1

        if len(self.past_trajectories) > 1 and len(self.opponent.past_trajectories) > 1:
            other_traj = self.opponent.past_trajectories[-2]
            previous_traj = self.past_trajectories[-2]

            self.progress_cnt += previous_traj.length
            self.progress_cost += previous_traj.relative_progress_costs[other_traj.number]

            self.out_bounds_cnt += previous_traj.out_bounds_cnt
            self.bounds_cost += previous_traj.bounds_cost
            
            self.proximity_cost += previous_traj.proximity_costs[other_traj.number]

        # negative means ahead
        is_ahead_conditions = ((self.previous_angle > self.opponent.previous_angle) and (self.laps_completed >= self.opponent.laps_completed))\
                           or (self.laps_completed > self.opponent.laps_completed)
        if is_ahead_conditions:
            # relative costs change sign, means bike got ahead
            if not self.is_ahead:
                self.pass_cnt += 1

            self.is_ahead = True
            self.ahead_cnt += 1

        else:
            self.is_ahead = False

    def new_choices(self, other_bike=None):
        """
        Generates new possible trajectories based on action choices.

        Parameters:
        - other_bike (Bicycle, optional): The opponent bicycle for collision checking.

        Returns:
        - None
        """

        # Precompute trajectories for visualization
        self.choice_trajectories = []
        self.action_choices = generate_combinations(self.action_lst, self.mpc_horizon)

        count = 0
        for action_sequence in self.action_choices:
            traj = Trajectory(bike=self, course=self.course, color=YELLOW)
            x_temp, y_temp, v_temp, phi_temp, b_temp = self.x, self.y, self.v, self.phi, self.b
            for action in action_sequence:
                acc = action[0] * ACCELERATION_INCREMENT
                steering = action[1] * STEERING_INCREMENT

                for _ in range(self.action_interval):
                    x_temp, y_temp, v_temp, phi_temp, b_temp = self.dynamics(acc, steering, x_temp, y_temp, v_temp, phi_temp, b_temp)
                    traj.add_point(x_temp, y_temp)

            traj.number = count
            self.choice_trajectories.append(traj)
            count += 1

        # relative costs
        if other_bike is not None and len(other_bike.choice_trajectories) > 0 and self.is_cost_populating:
            for traj in self.choice_trajectories:
                for other_traj in other_bike.choice_trajectories:
                    traj.relative_trajectory_sensing(other_traj)
                    traj.proximity_sensing(other_traj)
        print()

    def check_lap_completion(self):
        """
        Detects when the bicycle crosses the finish line to complete a lap.

        Returns:
        - None
        """

        current_angle = self.compute_angle()

        # Detect transition from just above 2π to just below 0
        if self.previous_angle > 1.8*pi:
            # Use to adjust costs of trajectories since angle of progress resets after 2pi
            self.is_crossing_finish = True
            if current_angle < 0.5 * pi:
                self.laps_completed += 1
                print(f"Bicycle {self.color} completed lap {self.laps_completed}")
        else:
            self.is_crossing_finish = False

        self.previous_angle = current_angle  # Update for next check
