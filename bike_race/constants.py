from math import radians
import numpy as np

# Main
IS_COST_DATA_CREATION_MODE = True

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
SEED = 41
IS_RANDOM_START=False
FRAME_RATE = 60

# Opponent Cost Weights
NUM_THETA_INTERVALS = 5
PROGRESS_RANGE = np.linspace(0.001, 1.0, NUM_THETA_INTERVALS)
BOUNDS_RANGE = np.linspace(0, 1.0, NUM_THETA_INTERVALS)
COLLISION_RANGE = np.linspace(0, 1.0, NUM_THETA_INTERVALS)

SPAWN_DICT = {'close_tail': (817, 728), # 3 car lengths, centered
              'far_tail': (919, 665), # 6 car lengths, centered
              'outside_edge': (871, 744), # 4 car lengths, 1/3 track width down
              'inside_edge': (836, 679)} # 4 car lengths, 1/3 track width up
ATTACKER_SPAWN_STATE = SPAWN_DICT['inside_edge']

# Course
# Data output path
RACE_DATA = "./data/race_stats.csv"
COST_DATA = "./data/cost_stats.csv"
ATTACKER_SPEED = 22.5
DEFENDER_SPEED = 15

# Whether costs are created via optimization of multiple objectives (vector)
# or weighted sum (scalar)
P1_IS_VECTOR_COST = False
P2_IS_VECTOR_COST = False

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
BIKE_SIZE = 40
# size for calculations, radial (width) and frontal (length) axes
LR = 1
LF = 1
# how close bike center points are in pixels to count as collision
COLLISION_RADIUS = 45

# Trajectory cost weights
progress_weight = 0
bounds_weight = 1
prox_weight = 1

PROXIMITY_SPREAD = 45
BOUNDS_SPREAD = 205

RELATIVE_PROGRESS_WEIGHT_1 = progress_weight
BOUNDS_WEIGHT_1 = bounds_weight
PROXIMITY_WEIGHT_1 = prox_weight

RELATIVE_PROGRESS_WEIGHT_2 = progress_weight
BOUNDS_WEIGHT_2 = bounds_weight
PROXIMITY_WEIGHT_2 = prox_weight

WEIGHTS_1 = [RELATIVE_PROGRESS_WEIGHT_1, BOUNDS_WEIGHT_1, PROXIMITY_WEIGHT_1]
WEIGHTS_2 = [RELATIVE_PROGRESS_WEIGHT_2, BOUNDS_WEIGHT_2, PROXIMITY_WEIGHT_2]