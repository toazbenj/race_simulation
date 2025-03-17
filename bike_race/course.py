import pygame
from math import atan2, pi, sqrt, ceil
from bicycle import Bicycle
import random
import csv
from constants import *


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
        # Note switched x,y phi for lead/follow switch
        # self.bike1 = Bicycle(self, x=x2, y=y2, phi=phi2, is_relative_cost=True, velocity_limit=15)
        # self.bike2 = Bicycle(self, x=x1, y=y1, phi=phi1, color=GREEN,
        #                      is_vector_cost=False, is_relative_cost=True, velocity_limit=10, opponent=self.bike1)
        # self.bike1.opponent = self.bike2

        self.bike1 = Bicycle(self, x=x1, y=y1, phi=phi1, is_relative_cost=True)
        self.bike2 = Bicycle(self, x=x2, y=y2, phi=phi2, color=GREEN,
                             is_vector_cost=True, is_relative_cost=True, velocity_limit=22.5, opponent=self.bike1)
        self.bike1.opponent = self.bike2

    def draw_button(self, screen, text, x, y, width, height, base_color, hover_color):
        """ Draws a button and returns True if clicked """
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

    def save_stats(self, race_number, seed):
        with open(WRITE_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            if race_number == 0:
                writer.writerow([' '])
                writer.writerow(["Race Number", 'Passes P1', 'Passes P2', 'Collisions',
                                 'Choices',
                                 'Proportion Ahead P1', 'Proportion Ahead P2',
                                 'Win P1', 'Win P2',
                                 'Progress P1', 'Progress P2',
                                 'Out of Bounds P1', 'Out of Bounds P2',
                                 'Progress Cost P1', 'Progress Cost P2',
                                 'Bounds Cost P1', 'Bounds Cost P2',
                                 'Proximity Cost P1', 'Proximity Cost P2',
                                 'Adjustment Count P2', f'Seed: {seed}'])

            try:
                p1_ahead = round(self.bike1.ahead_cnt/self.bike1.choice_cnt,2)
                p2_ahead = round(self.bike2.ahead_cnt/self.bike1.choice_cnt,2)
            except ZeroDivisionError:
                p1_ahead = 0
                p2_ahead = 0

            # P1 always ahead, pass count -1 since counts as a pass
            writer.writerow([race_number+1, self.bike1.pass_cnt, self.bike2.pass_cnt,
                             ceil(self.bike1.collision_cnt/200), self.bike1.choice_cnt,
                             p1_ahead, p2_ahead,
                             self.bike1.is_ahead, self.bike2.is_ahead,
                             round(self.bike1.progress_cnt), round(self.bike2.progress_cnt),
                             self.bike1.out_bounds_cnt, self.bike2.out_bounds_cnt,
                             round(self.bike1.progress_cost), round(self.bike2.progress_cost),
                             self.bike1.bounds_cost, self.bike2.bounds_cost,
                             self.bike1.proximity_cost, self.bike2.proximity_cost,
                             self.bike2.adjust_cnt])
