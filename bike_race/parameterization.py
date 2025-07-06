import numpy as np
from course import Course
import sembas_api as api
from constants import *
from matplotlib import pyplot as plt
import json

bounds = np.array(
    [
        [0.0, 10.0],
        [0.0, 10.0]    ]
)


def run_race(weights_1: list[float], weights_2: list[float], race: int, seed=41):
    # Initialize a new course with bikes in random positions
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    course = Course(
        center_x,
        center_y,
        weights_1,
        weights_2,
        race,
        inner_radius=INNER_RADIUS,
        outer_radius=OUTER_RADIUS,
        randomize_start=IS_RANDOM_START,
        seed=seed,
    )

    # search for collsions
    # i = 0
    # while i < RACE_DURATION and course.bike1.collision_cnt == 0:
    #     course.update()
    #     i += 1
    # return course.bike1.collision_cnt == 0

    # search for out of bounds
    # i = 0
    # while i < RACE_DURATION and course.bike2.out_bounds_cnt == 0:
    #     course.update()
    #     i += 1
    #     # print(f'bounds count: {course.bike2.out_bounds_cnt}')
    # return course.bike2.out_bounds_cnt == 0

    i = 0
    while i < RACE_DURATION and \
            (course.bike2.pass_cnt == 0
            and course.bike2.out_bounds_cnt == 0
            and course.bike1.collision_cnt == 0):

        course.update()
        i += 1
    return (
        course.bike2.pass_cnt == 0
        and course.bike2.out_bounds_cnt == 0
        and course.bike1.collision_cnt == 0
    )


def main():

    # (LOW, HIGH)
    session = api.SembasSession(bounds.T, plot_samples=False)

    requests = []
    results = []
    phase = []

    race = 0
    for i in range(100):
        print("=======================================================")
        print(f"Starting race {race}")

        cur_phase = session.expect_phase()
        x = session.receive_request()
        # result = run_race(x[:3], x[3:], race)

        result = run_race([1.0, 1.0, 1.0], [1.0, float(x[0]), float(x[1])], race)
        session.send_response(result)
        race += 1

        requests.append(x.tolist())
        results.append(result)
        phase.append(cur_phase)

        plt.pause(0.01)

    with open("results.json", "w") as f:
        # Unfortunately it isn't known which requests fall on a
        # boundary. That information is known on SEMBAS (in rust). However, you can
        # assume the points during the BE (boundary exploration phase) are
        # near the boundary. We can discuss how to look at the boundary
        # requests later.
        json.dump(
            {"requests": requests, "results": results, "phase": phase},
            f,
        )


main()
