from math import radians

# Main
# Screen dimensions
WIDTH, HEIGHT = 1400, 850
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
SEED = 42

# Course
# Data output path
WRITE_FILE = "../data2/test.csv"

# Bicycle
# Time step
DT = 0.05
# Control limits
STEERING_INCREMENT = radians(1)  # Increment for steering angle
ACCELERATION_INCREMENT = 3  # Increment for acceleration
STEER_LIMIT = radians(20)

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
