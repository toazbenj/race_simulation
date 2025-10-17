import numpy as np
from course import Course
from bicycle import Bicycle
import sembas_api as api
from constants import *
from matplotlib import pyplot as plt
import json

def write_data(description, requests, results, phase):
    print("Writing data")
    with open(SEMBAS_DATA, "r") as f:
        try:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
        except json.JSONDecodeError:
            existing_data = []

    data = {
        "description": description,
        "requests": requests,
        "results": results,
        "phase": phase,
    }
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
            randomize_start=False,
        )

        i = 0
        while i < RACE_DURATION and test_func(course.bike2, is_returning=False):
            course.update()
            i += 1
        return test_func(course.bike2, is_returning=True)

    return race_func

def run_until_phase(session: api.SembasSession, func, target_phase: str,) -> dict[str, list[tuple[tuple, bool]]]:
    """
    Returns the training data developed on a given pass of the
    SEMBAS algorithm.
    """
    result = {}
    
    session.expect_phase()
    prev_phase = None
    try:
        while session.prev_known_phase != target_phase:
            if session.prev_known_phase != prev_phase:
                result[session.prev_known_phase] = []
            x = session.receive_request()
            res = func([1.0, 1.0, 1.0], [float(x[0]), float(x[1]), float(x[2])])
            session.send_response(res)
            result[session.prev_known_phase].append((x, res))
            prev_phase = session.prev_known_phase
            session.expect_phase()
    except KeyboardInterrupt:
        return result

    return result

def run_test(description, race_func, session: api.SembasSession):
    requests = []
    results = []
    phase = []
    i = 0

    try:
        session.expect_phase()
        phase_dict = run_until_phase(session, race_func, "BE")

        for current_phase, entries in phase_dict.items():
            for request, result in entries:
                requests.append(request.tolist())
                results.append(result)
                phase.append(current_phase)

        while session.prev_known_phase != "GS":
            print("=======================================================")
            print(f"Starting race {i}")

            x = session.receive_request()

            result = race_func([1.0, 1.0, 1.0], [float(x[0]), float(x[1]), float(x[2])])
            session.send_response(result)

            requests.append(x.tolist())
            results.append(result)
            phase.append(session.prev_known_phase)
            session.expect_phase()
            print("Phase:", session.prev_known_phase)

            i += 1

        write_data(description, requests, results, phase)

        if session.prev_known_phase == "GS":
            print("Test completed")
        else:
            print("Test incomplete / ended early?")
    except:
        write_data(description, requests, results, phase)


def pass_test(bike, is_returning=False):
    if is_returning:
        return bike.pass_cnt == 1
    else:
        return bike.pass_cnt == 0


def bounds_test(bike, is_returning=False):
    return bike.out_bounds_cnt == 0


def collision_test(bike, is_returning=False):
    return bike.collision_cnt == 0


def main_multi_test():

    # (LOW, HIGH)
    session = api.SembasSession(BOUNDS.T, plot_samples=False)

    # full testing
    # scenario_lst = ["close_tail", "far_tail", "outside_edge", "inside_edge"]
    # test_func_lst = [pass_test, bounds_test, collision_test]

    # partial testing with scenarios/metrics that have boundaries (probably)
    scenario_lst = ["outside_edge"]
    test_func_lst = [collision_test]

    arg_lst = []
    # algorithm type - 0 or 1
    for algo in range(2):
        # target metric
        for test_func in test_func_lst:
            # physical spawn
            for scenario_key in scenario_lst:
                arg_lst.append([scenario_key, test_func, algo])

    print(arg_lst)
    # skipped bounds testing for weighted sum, starting at 8 for collision testing
    for i in range(1, len(arg_lst)):
        scenario_key, test_func, algo = arg_lst[i]
        description = f"Is Vector Cost: {algo}, Metric: {test_func}, Scenario: {scenario_key}"
        race_test = setup_race(SPAWN_DICT[scenario_key], test_func, algo)
        run_test(description, race_test, session)
        print(description)

try:
    main_multi_test()
except KeyboardInterrupt:
    print("Exiting due to keyboard interrupt")