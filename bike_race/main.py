import pygame
import sys
import random
from course import Course


pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1400, 850
WHITE = (255, 255, 255)

# Number of races to run
NUM_RACES = 100
RACE_DURATION = 1500  # Number of frames per race

# start line problems
# seed = 10

# passing
seed = 41

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Bicycle Dynamics Simulation")

    clock = pygame.time.Clock()

    random.seed(seed)
    seed_lst  = [random.randint(1, 1000) for _ in range(NUM_RACES)]

    for race in range(NUM_RACES):
        print(f"Starting Race {race + 1}")

        # Initialize a new course with bikes in random positions
        center_x, center_y = WIDTH // 2, HEIGHT // 2

        course = Course(center_x, center_y, inner_radius=250, outer_radius=400, randomize_start=True, seed=seed_lst[race])

        for _ in range(RACE_DURATION):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Update the simulation
            course.update()

            # Draw everything
            screen.fill(WHITE)
            course.draw(screen)
            pygame.display.flip()

            clock.tick(60)  # Limit frame rate

        course.save_stats(race)
        print(f"Race {race + 1} finished!")

if __name__ == "__main__":
    main()
