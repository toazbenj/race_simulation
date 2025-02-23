# import pygame
# from bicycle import Bicycle
# from math import radians
#
# # Colors
# WHITE = (255, 255, 255)
# GRAY = (169, 169, 169)
# BLACK = (0, 0, 0)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)
# DARK_GREEN = (0, 150, 0)
#
# class Course:
#     def __init__(self, center_x, center_y, outer_radius=300, inner_radius=125):
#         self.count = 0
#
#         self.center_x = center_x
#         self.center_y = center_y
#
#         self.outer_radius = outer_radius
#         self.inner_radius = inner_radius
#
#         self.bike1 = Bicycle(self, x=center_x, y=center_y + (inner_radius + outer_radius)/2, phi=radians(180), is_relative_cost=True, velocity_limit=15)
#         self.bike2 = Bicycle(self, x=center_x+100, y=center_y + (inner_radius + outer_radius)/2, phi=radians(180), color=GREEN,
#                              is_vector_cost=False, is_relative_cost=True, velocity_limit=22.5, opponent=self.bike1)
#
#         # self.bike1 = Bicycle(self, x=center_x + (inner_radius + outer_radius)/2, y=center_y, is_relative_cost=True, velocity_limit=15)
#         # self.bike2 = Bicycle(self, x=center_x + (inner_radius + outer_radius)/2, y=center_y-100, color=GREEN,
#         #                      is_vector_cost=True, is_relative_cost=True, velocity_limit=22.5, opponent=self.bike1)
#
#         # self.bike1 = Bicycle(self, x=center_x + (inner_radius + outer_radius)/2, y=center_y -100, is_relative_cost=True, velocity_limit=22.5)
#         # self.bike2 = Bicycle(self, x=center_x + (inner_radius + outer_radius)/2, y=center_y, color=GREEN,
#         #                      is_vector_cost=True, is_relative_cost=True, velocity_limit=15, opponent=self.bike1)
#
#     def draw(self, screen):
#         # Draw the racecourse
#         # Fill background
#         screen.fill(DARK_GREEN)
#
#         # Draw the outer circle
#         pygame.draw.circle(screen, GRAY, (self.center_x, self.center_y), self.outer_radius)
#
#         # Draw the inner circle (cutout)
#         pygame.draw.circle(screen, DARK_GREEN, (self.center_x, self.center_y), self.inner_radius)
#
#         # Draw boundaries
#         pygame.draw.circle(screen, BLACK, (self.center_x, self.center_y), self.outer_radius, 3)
#         pygame.draw.circle(screen, BLACK, (self.center_x, self.center_y), self.inner_radius, 3)
#
#         distance_cost1, bounds_cost1, collision_cost1, total_costs1 = self.bike1.get_costs()
#         distance_cost2, bounds_cost2, collision_cost2, total_costs2 = self.bike2.get_costs()
#
#         # display running costs
#
#         # Display running costs for each bike
#         font = pygame.font.Font(None, 36)  # Font and size
#
#         # Bike 1 costs
#         text_bike1 = font.render("Bike 1 Costs:", True, WHITE)
#         text_bike1_bounds = font.render(f"Bounds Cost: {int(bounds_cost1)}", True, WHITE)
#         text_bike1_distance = font.render(f"Distance Cost: {int(distance_cost1)}", True, WHITE)
#         text_bike1_collision = font.render(f"Collision Cost: {int(collision_cost1)}", True, WHITE)
#         text_bike1_total = font.render(f"Total Cost: {int(total_costs1)}", True, WHITE)
#
#         # Bike 2 costs
#         text_bike2 = font.render("Bike 2 Costs:", True, WHITE)
#         text_bike2_bounds = font.render(f"Bounds Cost: {int(bounds_cost2)}", True, WHITE)
#         text_bike2_distance = font.render(f"Distance Cost: {int(distance_cost2)}", True, WHITE)
#         text_bike2_collision = font.render(f"Collision Cost: {int(collision_cost1)}", True, WHITE)
#         text_bike2_total = font.render(f"Total Cost: {int(total_costs2)}", True, WHITE)
#
#         # Draw the costs on the right side of the screen
#         margin = 20
#         start_y = margin
#         screen_width = screen.get_width()
#
#         # Draw bike 1 costs
#         screen.blit(text_bike1, (screen_width - 300, start_y))
#         screen.blit(text_bike1_bounds, (screen_width - 300, start_y + 40))
#         screen.blit(text_bike1_distance, (screen_width - 300, start_y + 80))
#         screen.blit(text_bike1_collision, (screen_width - 300, start_y + 120))
#         screen.blit(text_bike1_total, (screen_width - 300, start_y + 160))
#
#         # Draw bike 2 costs
#         screen.blit(text_bike2, (screen_width - 300, start_y + 240))
#         screen.blit(text_bike2_bounds, (screen_width - 300, start_y + 280))
#         screen.blit(text_bike2_distance, (screen_width - 300, start_y + 320))
#         screen.blit(text_bike2_collision, (screen_width - 300, start_y + 360))
#         screen.blit(text_bike2_total, (screen_width - 300, start_y + 400))
#
#         # Draw the bicycle
#         self.bike1.draw(screen)
#         self.bike2.draw(screen)
#
#
#     def update(self):
#         self.bike1.update_choices(self.count, self.bike2)
#         self.bike2.update_choices(self.count, self.bike1)
#
#         self.bike1.update_action(self.count)
#         self.bike2.update_action(self.count)
#
#         self.count += 1


import pygame
from math import radians, atan2, pi, sqrt
from bicycle import Bicycle
import random
import csv

# Colors
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
WRITE_FILE = "race_stats.csv"


class Course:
    def __init__(self, center_x, center_y, outer_radius=300, inner_radius=125, randomize_start=False, seed=42):
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
            rand_x1 = center_x + random.uniform(-outer_radius, outer_radius)
            rand_y1 = center_y + random.uniform(-outer_radius, outer_radius)

            rand_x2 = center_x + random.uniform(-outer_radius, outer_radius)
            rand_y2 = center_y + random.uniform(-outer_radius, outer_radius)

            # Snap points to the centerline
            x1, y1 = self.snap_to_centerline(rand_x1, rand_y1)
            x2, y2 = self.snap_to_centerline(rand_x2, rand_y2)

        else:
            # Default start positions
            x1 = center_x
            y1 = center_y + self.centerline_radius

            x2 = center_x + 100
            y2 = center_y + self.centerline_radius

        phi1 = atan2(y1-center_y, x1-center_x) + pi/2
        phi2 = atan2(y2-center_y, x2-center_x) + pi/2

        # Initialize bikes facing directly down the track
        self.bike1 = Bicycle(self, x=x1, y=y1, phi=phi1, is_relative_cost=True, velocity_limit=15)
        self.bike2 = Bicycle(self, x=x2, y=y2, phi=phi2, color=GREEN,
                             is_vector_cost=False, is_relative_cost=True, velocity_limit=22.5, opponent=self.bike1)
        self.bike1.opponent = self.bike2

    def snap_to_centerline(self, x, y):
        """ Adjusts a point to the nearest position on the centerline. """
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
        screen.fill(DARK_GREEN)
        pygame.draw.circle(screen, GRAY, (self.center_x, self.center_y), self.outer_radius)
        pygame.draw.circle(screen, DARK_GREEN, (self.center_x, self.center_y), self.inner_radius)
        pygame.draw.circle(screen, BLACK, (self.center_x, self.center_y), self.outer_radius, 3)
        pygame.draw.circle(screen, BLACK, (self.center_x, self.center_y), self.inner_radius, 3)

        self.bike1.draw(screen)
        self.bike2.draw(screen)

    def update(self):
        self.bike1.update_choices(self.count, self.bike2)
        self.bike2.update_choices(self.count, self.bike1)

        self.bike1.update_action(self.count)
        self.bike2.update_action(self.count)

        self.count += 1

    def save_stats(self, race_number):
        with open(WRITE_FILE, mode='a') as file:
            writer = csv.writer(file)
            if race_number == 0:
                writer.writerow(["Race Number", 'Passes P1', 'Passes P2', 'Collisions',
                                 'Choices',
                                 'Proportion Ahead P1', 'Proportion Ahead P2',
                                 'Win P1', 'Win P2',
                                 'Progress P1', 'Progress P2',
                                 'Out of Bounds P1', 'Out of Bounds P2'])
            writer.writerow([race_number+1, self.bike1.pass_cnt, self.bike2.pass_cnt, self.bike1.collision_cnt,
                             self.bike1.choice_cnt,
                             round(self.bike1.ahead_cnt/self.bike1.choice_cnt,2), round(self.bike2.ahead_cnt/self.bike1.choice_cnt,2),
                             self.bike1.is_ahead, self.bike2.is_ahead,
                             round(self.bike1.progress_cnt), round(self.bike2.progress_cnt),
                             self.bike1.out_bounds_cnt, self.bike2.out_bounds_cnt])

