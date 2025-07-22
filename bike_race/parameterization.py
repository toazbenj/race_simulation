import numpy as np
from course import Course
from bicycle import Bicycle
import sembas_api as api
from constants import *
from matplotlib import pyplot as plt
import json

bounds = np.array(
    [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ]
)


def run_race(weights_1: list[float], weights_2: list[float], race: int, seed=SEED):
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
    # print(course.bike1.x, course.bike2.x, course.bike1.y, course.bike2.y)

    # search for collsions
    i = 0
    while i < RACE_DURATION and course.bike2.collision_cnt == 0:
        course.update()
        i += 1
    return course.bike2.collision_cnt == 0

    # search for out of bounds
    # i = 0
    # while i < RACE_DURATION and course.bike2.out_bounds_cnt == 0:
    #     course.update()
    #     i += 1
    #     # print(f'bounds count: {course.bike2.out_bounds_cnt}')
    # return course.bike2.out_bounds_cnt == 0


    # Exit condition at end of race time or for safety infraction (not at successful pass)
    # i = 0
    # while i < RACE_DURATION and \
    #         (course.bike2.out_bounds_cnt == 0) \
    #         and (course.bike2.collision_cnt == 0):

    #     course.update()
    #     i += 1

    # # Check if all goals satisfied (complete success)
    # return (
    #     course.bike2.pass_cnt == 1
    #     and course.bike2.out_bounds_cnt == 0
    #     and course.bike2.collision_cnt == 0
    # )


def main_old():

    # (LOW, HIGH)
    session = api.SembasSession(bounds.T, plot_samples=False)

    requests = []
    results = []
    phase = []

    race = 0

    try:
        for i in range(NUM_RACES):
            print("=======================================================")
            print(f"Starting race {race}")

            cur_phase = session.expect_phase()
            x = session.receive_request()

            result = run_race([1.0, 1.0, 1.0], [float(x[0]), float(x[1]), float(x[2])], race)
            session.send_response(result)
            race += 1

            requests.append(x.tolist())
            results.append(result)
            phase.append(cur_phase)

            plt.pause(0.01)

            # early save
            if i == NUM_RACES//2:
                with open(SEMBAS_DATA, "w") as f:
                    json.dump({"requests": requests, "results": results, "phase": phase}, f,)
    except:

        with open(SEMBAS_DATA, "w") as f:
            # Unfortunately it isn't known which requests fall on a
            # boundary. That information is known on SEMBAS (in rust). However, you can
            # assume the points during the BE (boundary exploration phase) are
            # near the boundary. We can discuss how to look at the boundary
            # requests later.
            json.dump(
                {"requests": requests, "results": results, "phase": phase},
                f,
            )

        print("Wrote data")
    finally:
        with open(SEMBAS_DATA, "w") as f:
            json.dump({"requests": requests, "results": results, "phase": phase}, f,)
        print("Wrote data")

def write_data(requests, results, phase):
    with open(SEMBAS_DATA, "r") as f:
        try:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
        except json.JSONDecodeError:
            existing_data = []

    data = {"requests": requests, "results": results, "phase": phase}
    existing_data.append(data)

    with open(SEMBAS_DATA, "w") as f:
        json.dump(existing_data, f, indent=2)


def setup_race(attacker_spawn_tup, test_func, is_vector_cost):
    def race_func(weights_1: list[float], weights_2: list[float]):
        # Initialize a new course with bikes in random positions
        course = Course(
            weights_1=weights_1,
            weights_2=weights_2,
            attacker_spawn_state=attacker_spawn_tup,
            is_player2_vector_cost=is_vector_cost,
            randomize_start=False)

        i = 0
        while i < RACE_DURATION and test_func(course.bike2, is_returning=False):
            course.update()
            i += 1
        return test_func(course.bike2, is_returning=True)
    return race_func

def run_test(race_func, session: api.SembasSession):
    requests = []
    results = []
    phase = []
    i = 0
    try:
        cur_phase = None
        while cur_phase != "NEXT":
            print("=======================================================")
            print(f"Starting race {i}")

            cur_phase = session.expect_phase()
            x = session.receive_request()

            result = race_func([1.0, 1.0, 1.0], [float(x[0]), float(x[1]), float(x[2])])
            session.send_response(result)

            requests.append(x.tolist())
            results.append(result)
            phase.append(cur_phase)

            i += 1

            # early save
            # if i == NUM_RACES//2:
            #     with open(SEMBAS_DATA, "w") as f:
            #         json.dump({"requests": requests, "results": results, "phase": phase}, f,)
        write_data(requests, results, phase)
        
    except Exception as e:
        print("Encountered unexpected error: {e}")
    finally:
        # with open(SEMBAS_DATA, "w") as f:
        #     # Unfortunately it isn't known which requests fall on a
        #     # boundary. That information is known on SEMBAS (in rust). However, you can
        #     # assume the points during the BE (boundary exploration phase) are
        #     # near the boundary. We can discuss how to look at the boundary
        #     # requests later.
        #     json.dump({"requests": requests, "results": results, "phase": phase}, f,)
        # print("Wrote data")
        write_data(requests, results, phase)        
    if cur_phase == "NEXT":
        print("Test completed")
    else:
        print("Test incomplete / ended early?")

def pass_test(bike, is_returning=False):
    if is_returning:
        return bike.pass_cnt == 1
    else:
        return bike.pass_cnt == 0

def bounds_test(bike, is_returning=False):
    return bike.out_of_bounds_cnt == 0

def collision_test(bike, is_returning=False):
    return bike.collision_cnt == 0

def main_multi_test():

    # (LOW, HIGH)
    session = api.SembasSession(bounds.T, plot_samples=False)

    scenario_lst = ['close_tail','far_tail','outside_edge','inside_edge']
    test_func_lst = [pass_test, bounds_test, collision_test]

    # physical spawn
    for scenario_key in scenario_lst:
        # target metric
        for test_func in test_func_lst:
            # algorithm type - 0 or 1
            for algo in range(2):
               race_test = setup_race(SPAWN_DICT[scenario_key], test_func, algo)     
               run_test(race_test, session)

    # # Define your first test
    # race_test_1 = setup_race(WIDTH//2, HEIGHT//2)
    # # Then run it. It will end when SEMBAS has found it's target number of boundary points
    # # This will trigger a new exploration to be established, and you can simply repeat.
    # run_test(race_test_1, session)

    # # repeat however many tests you have/need
    # race_test_2 = setup_race(WIDTH//3, HEIGHT//3)
    # run_test(race_test_2, session)


main_multi_test()
