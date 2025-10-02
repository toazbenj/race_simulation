import pygame
import sys
import random
import cv2
import numpy as np
from course import Course
from constants import *

CROP_X1, CROP_X2 = 250, WIDTH-250
CROP_Y1, CROP_Y2 =  HEIGHT//2, HEIGHT
CROP_W = CROP_X2 - CROP_X1
CROP_H = CROP_Y2 - CROP_Y1

video_name = 'videos/spawns/outside_edge.mp4'

def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Racing Simulation")

    clock = pygame.time.Clock()

    # --- Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_name,
                                fourcc,
                                FRAME_RATE,
                                (CROP_W, CROP_H))

    center_x, center_y = WIDTH // 2, HEIGHT // 2
    course = Course(center_x, center_y, inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS,
                    randomize_start=IS_RANDOM_START)
    try:
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
    except:
        # --- Finalize video ---
        video_writer.release()

    video_writer.release()

if __name__ == "__main__":
    main()
