import random
from course import Course
from constants import *
import numpy as np
import itertools

def main():
    random.seed(SEED)
    seed_lst = [random.randint(1, NUM_RACES) for _ in range(NUM_RACES)]    

    combinations = list(itertools.product(PROGRESS_RANGE, BOUNDS_RANGE, COLLISION_RANGE))
    weights_lst = np.array(combinations)
    weights = [RELATIVE_PROGRESS_WEIGHT_2, BOUNDS_WEIGHT_2, PROXIMITY_WEIGHT_2]

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
            course = Course(center_x, center_y, combination, [RELATIVE_PROGRESS_WEIGHT_2, BOUNDS_WEIGHT_2, PROXIMITY_WEIGHT_2], 
                            race, inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS,
                            randomize_start=IS_RANDOM_START, seed=seed_lst[race])

            for _ in range(RACE_DURATION):
                # Update the simulation
                course.update()

            course.save_stats()
            print(f"Race {race + 1} finished!")


if __name__ == "__main__":
    main()
