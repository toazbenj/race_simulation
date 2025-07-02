import random
from course import Course
from constants import *
import numpy as np
import itertools

def main():
    random.seed(SEED)
    seed_lst = [random.randint(1, 1000) for _ in range(NUM_THETA_INTERVALS**3)]

    weights_lst1 = []
    if IS_COST_DATA_CREATION_MODE:
        # Cartesian product using itertools
        combinations = list(itertools.product(PROGRESS_RANGE, BOUNDS_RANGE, COLLISION_RANGE))
        # Convert to NumPy array
        weights_lst1 = np.array(combinations)
    else:
        weights_1 = np.array([PROXIMITY_WEIGHT_1, BOUNDS_WEIGHT_1, RELATIVE_PROGRESS_WEIGHT_1])

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
            # Update the simulation
            course.update()

        course.save_stats(SEED)
        print(f"Race {race + 1} finished!")


if __name__ == "__main__":
    main()
