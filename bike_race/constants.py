from math import radians

# Main
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
NUM_RACES = 100
RACE_DURATION = 1500  # Number of frames per race
# Seed setting
SEED = 41
IS_RANDOM_START=True
FRAME_RATE = 60

# Course
# Data output path
WRITE_FILE = "../data2/test.csv"
ATTACKER_SPEED = 22.5
DEFENDER_SPEED = 15

# Relative costs based on performance relative to opponent
# (vs just progress relative to track)
P1_IS_RELATIVE_COST = True
P2_IS_RELATIVE_COST = True

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
# best combos: interval = 75, horizon = 2; interval = 40, mpc = 3
ACTION_INTERVAL = 75
MPC_HORIZON = 2
# Control inputs (acceleration, steering)
ACTION_LST = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
# how large the bike appears on screen
BIKE_SIZE = 20
# size for calculations, radial (width) and frontal (length) axes
LR = 1
LF = 1
# how close bike center points are in pixels to count as collision
COLLISION_RADIUS = 45


# Trajectory
# cost weights
COLLISION_WEIGHT = 750
DISTANCE_WEIGHT = -1/1000
BOUNDS_WEIGHT = 1000
RELATIVE_PROGRESS_WEIGHT = 100
# magnitude of collision badness
PROXIMITY_WEIGHT = 100
# lower numbers = more spread
DANGER_SPREAD = 5.5
