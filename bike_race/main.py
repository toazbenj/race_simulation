"""
Racing Simulation Game - Main Execution File

This script initializes and runs a racing simulation using Pygame. The simulation consists of multiple races where
bikes navigate a randomized circular course. The races can be manually skipped using an on-screen button.

Key Features:
- Utilizes the Pygame library for rendering and event handling.
- Generates random start positions using a seeded random number generator.
- Implements a "Skip Race" button to allow bypass of ongoing races.
- Ensures smooth execution with a fixed frame rate using Pygame's clock.

Modules Used:
- pygame: For graphics rendering and event handling.
- sys: To handle system exits.
- random: To generate seeded random values for race conditions.
- course: Custom module handling race course logic.
- constants: External file defining simulation parameters.

Entry Point:
- The script executes when run as the main module, launching the simulation.

"""

import pygame
import sys
import random
from course import Course
from constants import *
import numpy as np
import itertools

def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Racing Simulation")

    clock = pygame.time.Clock()

    random.seed(SEED)
    seed_lst = [random.randint(1, 1000) for _ in range(NUM_RACES)]

    weights_lst1 = []
    if IS_COST_DATA_CREATION_MODE:
        # Cartesian product using itertools
        combinations = list(itertools.product(PROGRESS_RANGE, BOUNDS_RANGE, COLLISION_RANGE))
        # Convert to NumPy array
        weights_lst1 = np.array(combinations)
    else:
        weights_1 = np.array([PROXIMITY_WEIGHT1, BOUNDS_WEIGHT_1, RELATIVE_PROGRESS_WEIGHT_1])

    weights_2 = np.array([PROXIMITY_WEIGHT_2, BOUNDS_WEIGHT_2, RELATIVE_PROGRESS_WEIGHT_2])

    for race in range(NUM_THETA_INTERVALS**3):
        if IS_COST_DATA_CREATION_MODE:
            weights_1 = weights_lst1[race]

        print(f"Starting Race {race + 1}")

        # Initialize a new course with bikes in random positions
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        course = Course(center_x, center_y, weights_1, weights_2, race,
                        inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS,
                        randomize_start=IS_RANDOM_START, seed=seed_lst[race])

        for _ in range(RACE_DURATION):
            skip_requested = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if course.draw_button(screen, "Skip Race", BUTTON_X, BUTTON_Y, BUTTON_W, BUTTON_H, BUTTON_COLOR,
                                          BUTTON_HOVER):
                        skip_requested = True

            if skip_requested:
                break

            # Update the simulation
            course.update()

            # Draw everything
            screen.fill(WHITE)
            course.draw(screen)

            # Draw Skip Button
            course.draw_button(screen, "Skip Race", BUTTON_X, BUTTON_Y, BUTTON_W, BUTTON_H, BUTTON_COLOR, BUTTON_HOVER)

            pygame.display.flip()
            clock.tick(FRAME_RATE)  # Limit frame rate

        course.save_stats(SEED)
        print(f"Race {race + 1} finished!")


if __name__ == "__main__":
    main()
