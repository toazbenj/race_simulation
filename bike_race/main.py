import pygame
import sys
import random
from course import Course
from constants import *


def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Racing Simulation")

    clock = pygame.time.Clock()

    random.seed(SEED)
    seed_lst = [random.randint(1, 1000) for _ in range(NUM_RACES)]

    button_x, button_y, button_w, button_h = 20, HEIGHT - 60, 150, 40  # Bottom left corner

    for race in range(NUM_RACES):
        print(f"Starting Race {race + 1}")

        # Initialize a new course with bikes in random positions
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        course = Course(center_x, center_y, inner_radius=250, outer_radius=400, randomize_start=True,
                        seed=seed_lst[race])

        for _ in range(RACE_DURATION):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Check if the skip button is clicked
            if course.draw_button(screen, "Skip Race", button_x, button_y, button_w, button_h, BUTTON_COLOR, BUTTON_HOVER):
                break  # Skip to next race

            # Update the simulation
            course.update()

            # Draw everything
            screen.fill(WHITE)
            course.draw(screen)

            # Draw Skip Button
            course.draw_button(screen, "Skip Race", button_x, button_y, button_w, button_h, BUTTON_COLOR, BUTTON_HOVER)

            pygame.display.flip()
            clock.tick(60)  # Limit frame rate

        course.save_stats(race, SEED)
        print(f"Race {race + 1} finished!")


if __name__ == "__main__":
    main()
