import random
from course import Course
from constants import *
import numpy as np
import itertools

import cProfile
import logging
import os
import atexit

PROFILE_PATH = os.path.expanduser('bike_race/logs/profile_vector2.prof')
logging.basicConfig(level=logging.INFO)

profiler = cProfile.Profile()
profiler.enable()

def main():
    random.seed(SEED)
    seed_lst = [random.randint(1, NUM_RACES) for _ in range(NUM_RACES)]    

    combinations = list(itertools.product(PROGRESS_RANGE, BOUNDS_RANGE, COLLISION_RANGE))
    weights_lst = np.array(combinations)
    weights = [RELATIVE_PROGRESS_WEIGHT_1, BOUNDS_WEIGHT_1, PROXIMITY_WEIGHT_1]

    # initialize to write headers, then overwrite later
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    course = Course(center_x, center_y, weights, weights, 
                    0, inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS,
                    randomize_start=IS_RANDOM_START, seed=seed_lst[0])
    course.write_race_stats_header(seed=SEED)
    course.write_cost_stats_header()

    for combination in weights_lst:
        for race in range(NUM_RACES):

            print(f"Starting Race {race + 1}")

            # Initialize a new course with bikes in random positions
            center_x, center_y = WIDTH // 2, HEIGHT // 2
            course = Course(center_x, center_y, weights, combination, 
                            race, inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS,
                            randomize_start=IS_RANDOM_START, seed=seed_lst[race])

            for _ in range(RACE_DURATION):
                # Update the simulation
                course.update()

            course.save_stats()
            print(f"Race {race + 1} finished!")

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
