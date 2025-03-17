from math import cos, sin, tan, atan2, sqrt, pi
import pygame
import numpy as np

from constants import *
from cost_adjust_cvx import find_adjusted_costs
from trajectory import Trajectory
from itertools import product

def generate_combinations(numbers, num_picks):
    """
    Generate all combinations of choices by picking `num_picks` times from the list `numbers`.

    Args:
        numbers (list): The list of numbers to pick from.
        num_picks (int): The number of times to pick.

    Returns:
        list of tuples: All combinations of length `num_picks`.
    """
    if not numbers or num_picks <= 0:
        return []

    # Use itertools.product to generate combinations
    combinations = list(product(numbers, repeat=num_picks))
    combinations = [list(element) for element in combinations]
    return combinations


class Bicycle:
    def __init__(self, course, x=300, y=300, v=5, color=BLUE, phi=radians(90), b=0, velocity_limit=15,
                 is_vector_cost=False, is_relative_cost=False, opponent=None):
        self.bicycle_size = 20
        self.color = color

        self.x = x
        self.y = y
        self.v = v
        self.phi = phi
        self.b = b

        self.lr = 1
        self.lf = 1

        self.a = 0
        self.steering_angle = 0
        self.velocity_limit = velocity_limit

        self.past_trajectories = []  # Store past positions
        self.choice_trajectories = [] # upcoming possible traj
        self.action_choices = [] # sequences of actions to create all possible trajectories
        self.chosen_action_sequence = [] # sequence of actions to create chosen trajectory

        # best combos: ai = 75, mpc = 2; ai = 40, mpc = 3
        self.action_interval = 75
        self.mpc_horizon = 1
        # acceleration, then steering
        self.action_lst = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

        self.course = course

        self.new_choices()
        self.is_vector_cost = is_vector_cost
        self.is_relative_cost = is_relative_cost
        self.opponent = opponent
        self.cost_arr = None

        # stats
        self.pass_cnt = 0
        self.collision_cnt = 0
        self.choice_cnt = 0
        self.ahead_cnt = 0
        self.is_ahead = True
        self.progress_cnt = 0
        self.out_bounds_cnt = 0
        self.adjust_cnt = 0
        self.collision_radius = 45

        self.progress_cost = 0
        self.bounds_cost = 0
        self.proximity_cost = 0

        # Lap tracking
        self.laps_completed = 0
        self.previous_angle = self.compute_angle()  # Initial angle
        self.is_crossing_finish = False

    def compute_angle(self):
        """
        Compute the bike's angle relative to the center of the track, but in a CLOCKWISE direction.
        - 0 radians is at the rightmost point.
        - -π/2 radians (or 3π/2) is at the top.
        - -π radians (or π) is at the leftmost point.
        - -3π/2 radians (or π/2) is at the bottom.
        - Angle **decreases clockwise**.
        """
        dx = self.x - self.course.center_x
        dy = self.y - self.course.center_y

        # Compute angle normally (CCW) using atan2
        angle = atan2(dy, dx)

        # Convert to clockwise by inverting and shifting
        # angle = -angle  # Invert the direction to make it clockwise

        # Normalize angle to the range [0, 2π]
        if angle < 0:
            angle += 2 * pi  # Ensures all angles stay positive in [0, 2π]

        return angle

    def dynamics(self, acc, steering, x_in, y_in, v_in, phi_in, b_in):
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

        return x_next, y_next, v_next, phi_next, b_next


    def draw(self, screen):
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
        if count % (self.action_interval * self.mpc_horizon) == 0:
            self.new_choices(other_bike)

    def update_action(self, count):
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
        size = len(self.action_lst)**self.mpc_horizon
        cost_arr = np.zeros((size, size))

        # absolute costs
        if not self.is_relative_cost:
            for i, traj in enumerate(trajectories):
                cost_row = np.zeros((1, size))
                cost_row[0, :] = traj.total_absolute_cost

                for other_traj in traj.intersecting_trajectories:
                    cost_row[0][other_traj.number] += traj.collision_weight

                cost_arr[i, :] = cost_row

        # relative costs

        else:
            for i, traj in enumerate(trajectories):
                cost_arr[i] = traj.total_relative_costs

                for other_traj in traj.intersecting_trajectories:
                    cost_arr[i][other_traj.number] += traj.collision_weight

        self.cost_arr = cost_arr
        # np.savez('B.npz', arr=self.cost_arr)

    def build_vector_arr(self, trajectories):
        size = len(self.action_lst) ** self.mpc_horizon
        safety_cost_arr = np.zeros((size, size))
        competitive_cost_arr = np.zeros((size, size))

        # absolute method
        if not self.is_relative_cost:
            for i, traj in enumerate(trajectories):
                cost_row_distance = np.zeros((1, size))
                cost_row_safety = np.zeros((1, size))

                cost_row_distance[0, :] = traj.distance_cost
                cost_row_safety[0, :] = traj.bounds_cost

                for other_traj in traj.intersecting_trajectories:
                    cost_row_safety[0][other_traj.number] += traj.collision_weight

                safety_cost_arr[i] = cost_row_safety
                competitive_cost_arr[i] = cost_row_distance

        # relative costs
        else:
            for i, traj in enumerate(trajectories):
                competitive_cost_arr[i] = traj.relative_arc_length_costs
                safety_cost_arr[i] = traj.trajectory_proximity_costs + traj.bounds_cost

                for other_traj in traj.intersecting_trajectories:
                    safety_cost_arr[i][other_traj.number] += traj.collision_weight

        E = find_adjusted_costs(competitive_cost_arr, safety_cost_arr, self.opponent.cost_arr.transpose())
        # E = find_adjusted_costs(safety_cost_arr, competitive_cost_arr, self.opponent.cost_arr.transpose())

        if E is None:
            print("no minima")
            self.cost_arr = competitive_cost_arr + safety_cost_arr
        else:
            print("adjustment success")
            self.cost_arr = competitive_cost_arr + E
            self.adjust_cnt += 1

        np.savez('../samples/A1.npz', arr=competitive_cost_arr)
        np.savez('../samples/A2.npz', arr=safety_cost_arr)

    def compute_action(self):
        if self.is_vector_cost:
            self.build_vector_arr(self.choice_trajectories)
        else:
            self.build_arr(self.choice_trajectories)

        action_index = np.argmin(np.max(self.cost_arr, axis=1))

        chosen_traj = self.choice_trajectories[action_index]
        chosen_traj.color = self.color
        chosen_traj.is_displaying = False
        chosen_traj.is_chosen = True

        # update costs of last trajectory after other players picked
        if len(self.past_trajectories) > 0:
            self.past_trajectories[-1].update()
        self.past_trajectories.append(chosen_traj)
        self.choice_trajectories.remove(chosen_traj)
        self.chosen_action_sequence = self.action_choices[action_index]

        self.update_stats()

    def update_stats(self):
        # update stats
        self.choice_cnt += 1

        if len(self.past_trajectories) > 1:
            previous_traj = self.past_trajectories[-2]
            self.progress_cnt += previous_traj.length
            self.progress_cost += previous_traj.relative_arc_length_outcome_cost
            self.proximity_cost += previous_traj.proximity_outcome_cost

            self.bounds_cost += previous_traj.bounds_cost
            if previous_traj.bounds_cost > 0:
                self.out_bounds_cnt += 1

            # print(previous_traj.relative_arc_length_outcome_cost, previous_traj.proximity_outcome_cost, previous_traj.bounds_cost)

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

        # check how far away the opponent is
        is_in_range = True
        if other_bike is not None:
            is_in_range = sqrt((self.x - other_bike.x) ** 2 + (self.y - other_bike.y) ** 2) < self.action_interval * self.mpc_horizon

        # allow traj to know possible collisions, absolute costs
        if other_bike is not None and len(other_bike.choice_trajectories) > 0 and is_in_range:
            for traj in self.choice_trajectories:
                # if traj.is_collision_checked:
                #     continue
                for other_traj in other_bike.choice_trajectories:
                    # if other_traj.is_collision_checked:
                    #     continue
                    other_traj.collision_checked = True
                    traj.collision_checked = True
                    traj.absolute_trajectory_sensing(other_traj, self.action_interval, self.mpc_horizon)
        # relative costs
        if other_bike is not None and len(other_bike.choice_trajectories) > 0:
            for traj in self.choice_trajectories:
                for other_traj in other_bike.choice_trajectories:
                    # if other_traj.is_relative_checked:
                    #     continue
                    traj.relative_trajectory_sensing(other_traj)

    def check_lap_completion(self):
        """
        Detects when the bike completes a lap by crossing 0 radians going clockwise.
        """
        current_angle = self.compute_angle()

        # Detect transition from just above 2π to just below 0
        if self.previous_angle > 1.8*pi:
            self.is_crossing_finish = True
            if current_angle < 0.5 * pi:
                self.laps_completed += 1
                print(f"Bicycle {self.color} completed lap {self.laps_completed}")
        else:
            self.is_crossing_finish = False

        self.previous_angle = current_angle  # Update for next check
