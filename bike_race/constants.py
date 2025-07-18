from math import radians
import numpy as np

# test

# Main
IS_COST_DATA_CREATION_MODE = False

# Screen dimensions
WIDTH, HEIGHT = 1400, 850

# Button sizes from bottom left corner
BUTTON_X = 20
BUTTON_Y = HEIGHT - 60
BUTTON_W = 150
BUTTON_H = 40
# Track dimensions
INNER_RADIUS=250
OUTER_RADIUS=400
# Colors
GRAY = (169, 169, 169)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 130, 80)
BUTTON_COLOR = (200, 0, 0)
BUTTON_HOVER = (255, 0, 0)

# Number of races to run
NUM_RACES = 50
RACE_DURATION = 1500  # Number of frames per race, base 1500

# Seed setting
SEED = 42
IS_RANDOM_START=True
FRAME_RATE = 60

# Opponent Cost Weights
NUM_THETA_INTERVALS = 10
PROGRESS_RANGE = np.linspace(1, 11, NUM_THETA_INTERVALS)
BOUNDS_RANGE = np.linspace(1, 11, NUM_THETA_INTERVALS)
COLLISION_RANGE = np.linspace(1, 11, NUM_THETA_INTERVALS)

# Course
# Data output path
RACE_DATA = "../data/race_stats.csv"
COST_DATA = "../data/cost_stats.csv"

SEMBAS_DATA = "../data/vector_collision_test.json"
# SEMBAS_DATA = "../data/vector_test.json"
# SEMBAS_DATA = 'test.json'

ATTACKER_SPEED = 22.5
DEFENDER_SPEED = 15

# Whether costs are created via optimization of multiple objectives (vector)
# or weighted sum (scalar)
P1_IS_VECTOR_COST = False
P2_IS_VECTOR_COST = True

# Bicycle
# Time step
DT = 0.05
# Control limits, amount of control effort per choice
STEERING_INCREMENT = radians(1)
ACCELERATION_INCREMENT = 3
STEER_LIMIT = radians(20)
# best combos: interval = 70, horizon = 1;
# interval = 50, horizon = 2; interval = 40, mpc = 3
ACTION_INTERVAL = 50
MPC_HORIZON = 1
# Control inputs (acceleration, steering)
ACTION_LST = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
# how large the bike appears on screen
BIKE_SIZE = 20
# size for calculations, radial (width) and frontal (length) axes
LR = 1
LF = 1
# how close bike center points are in pixels to count as collision
COLLISION_RADIUS = 22.5
PROXIMITY_SPREAD = 21
BOUNDS_SPREAD = 205

# Trajectory cost weights

# point 38 - false positives - scalar_match_test - complete success
# progress_weight = 0.53
# bounds_weight = 0.49
# prox_weight = 0.46

# point 44 - false positives - scalar_match_test
# progress_weight = 0.58
# bounds_weight = 0.54
# prox_weight = 0.47

# point 20 - false positives - scalar_match_test
# progress_weight = 0.48
# bounds_weight = 0.46
# prox_weight = 0.50

progress_weight = 0.74
bounds_weight = 0.20
prox_weight = 0.0



RELATIVE_PROGRESS_WEIGHT_1 = 1.0
BOUNDS_WEIGHT_1 = 1.0
PROXIMITY_WEIGHT_1 = 1.0

RELATIVE_PROGRESS_WEIGHT_2 = progress_weight
BOUNDS_WEIGHT_2 = bounds_weight
PROXIMITY_WEIGHT_2 = prox_weight
