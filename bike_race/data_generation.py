import random
from course import Course
from constants import *
import numpy as np
import itertools

import cProfile
import logging
import os
import atexit
import pygame

logging.basicConfig(level=logging.INFO)

profiler = cProfile.Profile()
profiler.enable()

def main():

    combinations = list(itertools.product(PROGRESS_RANGE, BOUNDS_RANGE, COLLISION_RANGE))
    weights_lst = np.array(combinations)

    # np.set_printoptions(precision=3, suppress=True)
    # print(len(combinations))
    # print(weights_lst)

    # graphics
    # pygame.init()
    # screen = pygame.display.set_mode((WIDTH, HEIGHT))
    # pygame.display.set_caption("Racing Simulation")
    # clock = pygame.time.Clock()

    cnt = 0
    for is_attacker_vector_cost in [False, True]:

        # initialize to write headers, then overwrite later
        course = Course()
        course.write_race_stats_header(seed=SEED)
        course.write_cost_stats_header()

        for scenario in SPAWN_DICT.values():
            for combination in weights_lst:

                print(f"Starting Race {cnt + 1}")

                course = Course(inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS,
                                is_player2_vector_cost=is_attacker_vector_cost,
                                weights_2=combination,
                                attacker_spawn_state=scenario, race_number=cnt)

                for _ in range(RACE_DURATION):
                    # Update the simulation
                    course.update()

                    # graphics
                    # screen.fill(WHITE)
                    # course.draw(screen)
                    # course.draw_button(screen, "Skip Race", BUTTON_X, BUTTON_Y, BUTTON_W, BUTTON_H, BUTTON_COLOR, BUTTON_HOVER)
                    # pygame.display.flip()
                    # clock.tick(FRAME_RATE)  # Limit frame rate

                course.save_race_stats()
                print(f"Race {cnt + 1} finished!")
                cnt += 1

def save_profile():
    profiler.disable()
    profiler.dump_stats(PROFILE_PATH)
    if os.path.exists(PROFILE_PATH):
        logging.info(f"Profile successfully saved to {PROFILE_PATH}")
    else:
        logging.error(f"Profile not saved. Expected at {PROFILE_PATH}")

atexit.register(save_profile)

if __name__ == "__main__":
    main()
