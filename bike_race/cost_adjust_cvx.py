"""
Cost Adjustment and Potential Function Optimization

This module provides functions for adjusting cost matrices to enforce potential function constraints 
and ensure a correct global minimum in decision-making models.

Key Features:
- **Cost Adjustment:** Modifies cost matrices to satisfy potential function constraints.
- **Potential Function Calculation:** Computes a potential function from cost matrices.
- **Exact Potential Verification:** Validates whether a potential function meets exact potential conditions.
- **Global Minimum Enforcement Check:** Ensures a global minimum is correctly enforced in cost matrices.
- **Pareto Optimality Filtering:** Identifies Pareto-optimal decisions that are not strictly dominated.
- **Convex Optimization with CVXPY:** Solves optimization problems to find minimal cost adjustments.

Modules Used:
- numpy: For numerical operations and matrix calculations.
- cvxpy: To perform convex optimization for error minimization.

Functions:
- `cost_adjustment(A, B, global_min_position)`: Adjusts cost matrices to satisfy exact potential constraints.
- `potential_function(A, B, global_min_position)`: Computes a potential function from cost matrices.
- `is_valid_exact_potential(A, B, phi)`: Checks if a given potential function satisfies exact conditions.
- `is_global_min_enforced(phi, global_min_position)`: Ensures the global minimum is correctly enforced.
- `pareto_optimal(A1, B1, column)`: Identifies Pareto-optimal rows in two cost matrices.
- `find_adjusted_costs(A1, B1, C2)`: Determines the best cost adjustments using convex optimization.

Entry Point:
- This module can be imported for cost adjustment in decision-making algorithms or used independently for testing.
"""

import numpy as np
import cvxpy as cp

def cost_adjustment(A1, D2, global_min_position):
    """
    Adjusts the cost matrices to enforce potential function constraints through convex optimization.

    Parameters:
    - A1 (np.ndarray): Cost matrix for player 1.
    - D2 (np.ndarray): Cost matrix for player 2.
    - global_min_position (tuple[int, int]): Coordinates (row, column) of the desired global minimum.

    Returns:
    - np.ndarray: Adjusted error matrix E that satisfies the potential function conditions.
    """

    # Compute initial potential function
    phi_initial = potential_function(A1, D2, global_min_position)

    if is_valid_exact_potential(A1, D2, phi_initial) and \
            is_global_min_enforced(phi_initial, global_min_position):
        E = np.zeros_like(A1)
        return E

    # Convex optimization to find Ea
    m, n = A1.shape
    E = cp.Variable((m, n))
    phi = cp.Variable((m, n))
    A_prime = A1 + E
    constraints = []

    # Constraint 1: Ensure global minimum position is zero
    constraints.append(phi[global_min_position[0], global_min_position[1]] == 0)

    # Constraint 2: Enforce non-negativity
    epsilon = 1e-6
    for k in range(m):
        for j in range(n):
            if (k, j) != tuple(global_min_position):
                constraints.append(phi[k, j] >= epsilon)

    # Constraint 3: Ensure exact potential condition
    for k in range(1, m):
        for l in range(n):
            delta_A = A_prime[k , l] - A_prime[k-1, l]
            delta_phi = phi[k, l] - phi[k-1, l]
            constraints.append(delta_A == delta_phi)

    for k in range(m):
        for l in range(1, n):
            delta_B = D2[k, l] - D2[k, l - 1]
            delta_phi = phi[k, l] - phi[k, l - 1]
            constraints.append(delta_B == delta_phi)

    # Solve optimization problem
    objective = cp.Minimize(cp.norm(E, 'fro'))
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.SCS, max_iters=50000, eps=1e-6, verbose=False)
    return E.value


def potential_function(A, B, global_min_position):
    """
    Computes a potential function for given cost matrices.

    Parameters:
    - A (np.ndarray): Cost matrix for player 1.
    - B (np.ndarray): Cost matrix for player 2.
    - global_min_position (tuple[int, int]): Coordinates (row, column) of the desired global minimum.

    Returns:
    - np.ndarray: Computed potential function matrix.
    """
    m, n = A.shape
    phi = np.zeros((m, n))

    for i in range(1, m):
        phi[i, 0] = phi[i - 1, 0] + A[i, 0] - A[i - 1, 0]

    for j in range(1, n):
        phi[0, j] = phi[0, j - 1] + B[0, j] - B[0, j - 1]

    for i in range(1, m):
        for j in range(1, n):
            phi[i, j] = (phi[i - 1, j] + A[i, j] - A[i - 1, j] +
                         phi[i, j - 1] + B[i, j] - B[i, j - 1]) / 2

    return phi - phi[global_min_position[0], global_min_position[1]]


def is_valid_exact_potential(A, B, phi):
    """
    Checks if the given potential function satisfies the exact potential condition.

    Parameters:
    - A (np.ndarray): Cost matrix for player 1.
    - B (np.ndarray): Cost matrix for player 2.
    - phi (np.ndarray): Potential function matrix.

    Returns:
    - bool: True if the exact potential condition is satisfied, False otherwise.
    """

    m, n = A.shape
    epsilon = 1e-6

    for i in range(1, m):
        for j in range(n):
            if not np.isclose((A[i, j] - A[i - 1, j]), (phi[i, j] - phi[i - 1, j]), atol=epsilon):
                return False

    for i in range(m):
        for j in range(1, n):
            if not np.isclose((B[i, j] - B[i, j-1]), (phi[i, j] - phi[i, j-1]), atol=epsilon):
                return False

    return True


def is_global_min_enforced(phi, global_min_position):
    """
    Checks if the global minimum is correctly enforced in the potential function.

    Parameters:
    - phi (np.ndarray): Potential function matrix.
    - global_min_position (tuple[int, int]): Coordinates (row, column) of the global minimum.

    Returns:
    - bool: True if the global minimum is enforced, False otherwise.
    """

    m, n = phi.shape
    if phi[global_min_position[0], global_min_position[1]] != 0:
        return False

    for i in range(m):
        for j in range(n):
            if (i, j) != tuple(global_min_position) and phi[i, j] <= 0:
                return False

    return True


def pareto_optimal(A1, B1, C1, column):
    """
    Identifies Pareto-optimal rows in two cost matrices where no other row is strictly better.

    Parameters:
    - A1 (np.ndarray): Cost matrix for player 1.
    - B1 (np.ndarray): Cost matrix for player 2.
    - column (int): The column index for evaluating Pareto optimality.

    Returns:
    - np.ndarray: Indices of Pareto-optimal rows.
    """

    num_rows = A1.shape[0]
    pareto_indices = []

    for i in range(num_rows):
        dominated = False
        for j in range(num_rows):
            if i != j:
                # Check if row j dominates row i

                # if (A1[j][column] <= A1[i][column] and B1[j][column] <= B1[i][column]) and (
                #         A1[j][column] < A1[i][column] or B1[j][column] < B1[i][column]):
                #     dominated = True
                #     break

                dominates = (
                        A1[j][column] <= A1[i][column] and
                        B1[j][column] <= B1[i][column] and
                        C1[j][column] <= C1[i][column]
                )
                strictly_better = (
                        A1[j][column] < A1[i][column] or
                        B1[j][column] < B1[i][column] or
                        C1[j][column] < C1[i][column]
                )

                # print(f"dominates: {dominated}, strictly better: {strictly_better}")

                if dominates and strictly_better:
                    dominated = True
                    break
        if not dominated:
            pareto_indices.append(i)

    return np.array(pareto_indices)


def find_adjusted_costs(A1, B1, C1, D2):
    """
    Determines the best cost adjustment matrix E using convex optimization.

    Parameters:
    - A1 (np.ndarray): Cost matrix for player 1.
    - B1 (np.ndarray): Cost matrix for player 1.
    - C1 (np.ndarray): Cost matrix for the second player's strategy.

    Returns:
    - np.ndarray: Adjusted cost matrix E, or None if no valid adjustment is found.
    """

    player2_sec_policy = np.argmin(np.max(D2, axis=0), axis=0)

    # find worst case safety actions and avoid
    safe_row_indices = np.where((~np.any(B1 == np.max(B1), axis=1)) &
                                (~np.any(A1 == np.max(A1), axis=1)) &
                                (~np.any(C1 == np.max(C1), axis=1)))[0]

    # add operation that selects only pareto optimal indices
    pareto_indices = pareto_optimal(A1, B1, C1, player2_sec_policy)
    # print(pareto_indices)
    pareto_safe_indices = np.intersect1d(safe_row_indices, pareto_indices)
    print("Pareto safe indicies: " + str(pareto_safe_indices))

    # find error matrices to make each combination of indices the global min of potential function
    E_star = np.ones_like(A1) * np.inf
    for i in pareto_safe_indices:
        min_position = (i, player2_sec_policy)
        E = cost_adjustment(A1, D2, min_position)

        if E is not None:
            phi = potential_function(E + A1, D2, min_position)
            is_min = is_global_min_enforced(phi, min_position)
            is_exact = is_valid_exact_potential(A1 + E, D2, phi)

            # print(is_min, is_exact)
            if is_min and is_exact and (np.linalg.norm(E) < np.linalg.norm(E_star)):

                player1_sec = np.argmin(np.max(A1 + E, axis=1))

                print("Minimum position: ", min_position)
                print('P2 Sec: ', player1_sec)

                if min_position[0] == player1_sec:
                    E_star = E

    if np.any(np.isinf(E_star)):
        return None
    else:
        # print(E_star+A1)
        return E_star


if __name__ == '__main__':
    # ======= Test Matrices =======
    # A1 = np.array([[0, 1, 2],
    #                [-1, 0, 1],
    #                [-2, -1, 0]])
    # A2 = np.array([[0, 1, 2],
    #                [1, 2, 3],
    #                [2, 3, 4]])
    #
    # B = -A1+ 2*A2

    # size = 9
    # A1 = np.random.randint(0, 10, (size, size))  # Random integers from 0 to 9
    # B1 = np.random.randint(0, 10, (size, size))
    # A2 = np.random.randint(0, 10, (size, size))

    # A1 = np.array([[3, 1, 4, 1, 5, 8, 1 ,5, 7],
    #              [5, 3, 3, 2, 5, 8, 7, 1, 2],
    #              [5, 4, 6, 1, 0 ,7, 8, 5, 2],
    #              [0, 8, 2, 7, 6, 8, 6, 6, 6],
    #              [9, 7, 4, 5, 2, 4, 9, 4, 3],
    #              [0, 2, 7, 6, 4, 8, 5, 4, 4],
    #              [4, 9, 9, 5, 4, 6, 7, 9, 1],
    #              [9, 3, 4, 1, 6, 2, 2, 6, 7],
    #              [8, 6, 9, 7, 9, 7, 6, 1, 8]])
    #
    # A2 = np.array([[2, 7, 2, 6 ,1 ,3 ,5 ,9, 5],
    #              [1, 6, 9 ,0 ,1 ,2, 3, 8, 7],
    #              [3, 6, 8, 5, 5, 7, 3, 9, 2],
    #              [3, 0, 4, 0, 1, 3, 3, 7, 9],
    #              [4, 2, 5, 2, 7 ,9 ,1, 5, 9],
    #              [1 ,1 ,0 ,1, 7, 3, 0, 8, 6],
    #              [9, 3, 8, 3, 9, 0, 0, 5, 3],
    #              [8 ,5 ,8, 9, 5, 0, 8, 1, 4],
    #              [4, 5, 9, 4, 9, 6, 9, 1, 7]])
    #
    # B = np.array( [[9, 3, 5, 1, 3, 6, 6, 5, 5],
    #              [8, 7, 6, 3, 9, 0, 9, 3, 4],
    #              [8, 3, 1, 6, 4, 7, 8, 4, 6],
    #              [6, 5, 0, 5, 9, 9, 9, 3, 5],
    #              [0, 6, 7, 8, 7, 0, 2, 4, 0],
    #              [8, 9, 0 ,1, 3, 6, 4, 9, 1],
    #              [6, 0, 3, 7, 7, 5, 3, 1, 3],
    #              [9, 0, 0, 7, 7, 8, 1, 9, 9],
    #              [6, 3, 8, 2, 5, 7, 9, 4, 4]])

    A1_load = np.load('../samples/A1.npz')
    A2_load = np.load('../samples/A1.npz')
    B_load = np.load('../samples/A1.npz')

    # Extract the array
    A1 = A1_load['arr']
    A2 = A2_load['arr']
    B = B_load['arr']
    E = find_adjusted_costs(A1, A2, B)

    if E is None:
        print('None')

    else:
        A_prime = A1 + E
        player1_sec = np.argmin(np.max(A_prime, axis=1))
        player2_sec = np.argmin(np.max(B.transpose(), axis=1))

        min_position = (player1_sec, player2_sec)
        phi = potential_function(A_prime, B, min_position)

        print("Error:")
        print(E)
        print("Player 1 A_prime:")
        print(A_prime)
        print("Potential Function:")
        print(phi)
        print("Global Min:", int(min_position[0]), int(min_position[1]))
        print("Global Minimum Enforced:", is_global_min_enforced(phi, min_position))
        print("Exact Potential:", is_valid_exact_potential(A_prime, B, phi))
