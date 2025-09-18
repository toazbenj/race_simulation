import pygame
import sys
import random
import cv2
import numpy as np
from course import Course
from constants import *

CROP_X1, CROP_X2 = 250, WIDTH-250
CROP_Y1, CROP_Y2 = 0, HEIGHT
CROP_W = CROP_X2 - CROP_X1
CROP_H = CROP_Y2 - CROP_Y1

def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Racing Simulation")

    clock = pygame.time.Clock()

    random.seed(SEED)
    seed_lst = [random.randint(1, 1000) for _ in range(NUM_RACES)]

    # race_lst = [10,18,47,50,93,94]
    # race_lst = [3,4,10,11,12,13,16,17,18,20,23,26,27,28,32,40,41,46,47,
    # 50,55,57,64,66,69,72,83,84,89,90,91,93,94,95]
    race_lst = [3,10,12,27,32,40,46,66,69,72,89,91,94]

    for race in race_lst:
        print(f"Starting Race {race}")

        # --- Setup Video Writer ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(f"videos/r{race}n.mp4",
                                    fourcc,
                                    FRAME_RATE,
                                    (CROP_W, CROP_H))

        center_x, center_y = WIDTH // 2, HEIGHT // 2
        course = Course(center_x, center_y, inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS,
                        randomize_start=IS_RANDOM_START, seed=seed_lst[race])

        for _ in range(RACE_DURATION):
            skip_requested = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    video_writer.release()
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if course.draw_button(screen, "Skip Race", BUTTON_X, BUTTON_Y, BUTTON_W, BUTTON_H, BUTTON_COLOR, BUTTON_HOVER):
                        skip_requested = True

            if skip_requested:
                break

            # Update simulation
            course.update()

            # Draw
            screen.fill(WHITE)
            course.draw(screen)
            course.draw_button(screen, "Skip Race", BUTTON_X, BUTTON_Y, BUTTON_W, BUTTON_H, BUTTON_COLOR, BUTTON_HOVER)
            pygame.display.flip()

            # --- Capture frame ---
            frame = pygame.surfarray.array3d(screen)        # (width, height, 3)
            frame = np.transpose(frame, (1, 0, 2))          # Swap axes to (height, width, 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB → BGR
            frame = frame[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
            video_writer.write(frame)

            clock.tick(FRAME_RATE)

        # --- Finalize video ---
        video_writer.release()

if __name__ == "__main__":
    main()
