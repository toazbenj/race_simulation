import pygame
import sys
import random
from course import Course

pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1400, 850
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BUTTON_COLOR = (200, 0, 0)
BUTTON_HOVER = (255, 0, 0)

# Number of races to run
NUM_RACES = 100
RACE_DURATION = 1500  # Number of frames per race

# Seed setting
seed = 41


def draw_button(screen, text, x, y, width, height, base_color, hover_color):
    """ Draws a button and returns True if clicked """
    mouse_pos = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    # Change color on hover
    color = hover_color if x < mouse_pos[0] < x + width and y < mouse_pos[1] < y + height else base_color

    pygame.draw.rect(screen, color, (x, y, width, height), border_radius=10)

    # Render text
    font = pygame.font.Font(None, 36)
    text_surf = font.render(text, True, WHITE)
    text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surf, text_rect)

    # Check for click
    if click[0] == 1 and x < mouse_pos[0] < x + width and y < mouse_pos[1] < y + height:
        return True  # Button clicked

    return False


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Bicycle Dynamics Simulation")

    clock = pygame.time.Clock()

    random.seed(seed)
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
            if draw_button(screen, "Skip Race", button_x, button_y, button_w, button_h, BUTTON_COLOR, BUTTON_HOVER):
                break  # Skip to next race

            # Update the simulation
            course.update()

            # Draw everything
            screen.fill(WHITE)
            course.draw(screen)

            # Draw Skip Button
            draw_button(screen, "Skip Race", button_x, button_y, button_w, button_h, BUTTON_COLOR, BUTTON_HOVER)

            pygame.display.flip()
            clock.tick(60)  # Limit frame rate

        course.save_stats(race, seed)
        print(f"Race {race + 1} finished!")


if __name__ == "__main__":
    main()
