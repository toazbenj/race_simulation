import numpy as np

def pareto_optimal(A1, B1, column):
    """
    Find Pareto-optimal rows where no other row is strictly better in both A1 and B1.
    """
    num_rows = A1.shape[0]
    pareto_indices = []

    for i in range(num_rows):
        dominated = False
        for j in range(num_rows):
            if i != j:
                # Check if row j dominates row i
                if A1[j][column] <= A1[i][column] and B1[j][column] <= B1[i][column] and (
                        A1[j][column] < A1[i][column] or B1[j][column] < B1[i][column]):
                    dominated = True
                    break
        if not dominated:
            pareto_indices.append(i)

    return np.array(pareto_indices)

# def pareto_optimal(A1, B1, column):
#     """
#     Find Pareto-optimal rows where no other row is strictly better in both A1 and B1
#     for a specific column.
#
#     A1: np.ndarray of shape (n, m) - first criteria matrix
#     B1: np.ndarray of shape (n, m) - second criteria matrix
#     column: int - the column index to compare
#
#     Returns:
#     np.array of indices that are Pareto-optimal.
#     """
#     num_rows = A1.shape[0]
#     pareto_indices = []
#
#     for i in range(num_rows):
#         dominated = False
#         for j in range(num_rows):
#             if i != j:
#                 # Check if row j dominates row i in the specified column
#                 if ((A1[j, column] <= A1[i, column] and B1[j, column] <= B1[i, column]) and
#                         (A1[j, column] < A1[i, column] or B1[j, column] < B1[i, column])):
#                     dominated = True
#                     break
#         if not dominated:
#             pareto_indices.append(i)
#
#     return np.array(pareto_indices)


# Example 1: Single-column matrices (works as expected)
# A1 = np.array([[38.19225],
#                [36.66295],
#                [37.43710],
#                [32.78521],
#                [28.89867],
#                [30.08999],
#                [32.78521],
#                [28.89867],
#                [30.08999]])
#
# B1 = np.array([[0],
#                [0],
#                [7000],
#                [0],
#                [0],
#                [22000],
#                [0],
#                [0],
#                [22000]])

A1 = np.array([[31.39284,31.44590,31.13272,37.52103,38.19225,37.41867,37.52103,38.19225,37.41867],
[29.86354,29.91659,29.60341,35.99172,36.66295,35.88936,35.99172,36.66295,35.88936],
[30.63769,30.69074,30.37757,36.76587,37.43710,36.66351,36.76587,37.43710,36.66351],
[25.98580,26.03885,25.72568,32.11398,32.78521,32.01162,32.11398,32.78521,32.01162],
[22.09926,22.15232,21.83914,28.22745,28.89867,28.12509,28.22745,28.89867,28.12509],
[23.29058,23.34363,23.03046,29.41876,30.08999,29.31640,29.41876,30.08999,29.31640],
[25.98580,26.03885,25.72568,32.11398,32.78521,32.01162,32.11398,32.78521,32.01162],
[22.09926,22.15232,21.83914,28.22745,28.89867,28.12509,28.22745,28.89867,28.12509],
[23.29058,23.34363,23.03046,29.41876,30.08999,29.31640,29.41876,30.08999,29.31640]])

B1 = np.array([[0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000],
[0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000],
[7000.00000,7000.00000,7000.00000,7000.00000,7000.00000,7000.00000,7000.00000,7000.00000,7000.00000],
[0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000],
[0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000],
[22000.00000,22000.00000,22000.00000,22000.00000,22000.00000,22000.00000,22000.00000,22000.00000,22000.00000],
[0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000],
[0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000],
[22000.00000,22000.00000,22000.00000,22000.00000,22000.00000,22000.00000,22000.00000,22000.00000,22000.00000]])

print("Single-column test:", pareto_optimal(A1, B1, 4))  # Expected correct output
