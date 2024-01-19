import pygame

# Initialize Pygame
pygame.init()

# Window properties
window_width = 700
window_height = 600
screen_width = 600
screen_height = 600
number_cells = 6

# Create window and surfaces
screen = pygame.display.set_mode((window_width, window_height))
surface1 = pygame.Surface((screen_width, screen_height))
surface2 = pygame.Surface((window_width - screen_width, screen_height))

# Load images
bg_image = pygame.image.load('bg_02_h.png')
truck_image = pygame.image.load('truck.png')

# Transform and scale images
#bg_image = pygame.transform.scale(bg_image, (screen_width, screen_height))
truck_image = pygame.transform.scale(truck_image, ((screen_width // number_cells) * 7 // 10,) * 2)

# Fill surfaces
surface1.blit(bg_image, (0, 0))
surface2.fill((140, 143, 194))

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill background with surfaces
    screen.blit(surface1, (0, 0))
    screen.blit(surface2, (screen_width, 0))

    # Draw grid lines
    pix_square_size = screen_width // number_cells
    for x in range(number_cells):
        pygame.draw.line(screen, (29, 191, 219), (0, pix_square_size * x), (screen_width, pix_square_size * x), 3)
        pygame.draw.line(screen, (29, 191, 219), (pix_square_size * x, 0), (pix_square_size * x, screen_height), 3)

    # Additional lines at the extremities of the grid
    pygame.draw.line(screen, (29, 191, 219), (screen_width, screen_width // number_cells), (screen_width, screen_width), 3)
    pygame.draw.line(screen, (29, 191, 219), (0, screen_width), (screen_width, screen_width), 6)

    # Place agent image (truck)
    #screen.blit(truck_image, (int((screen_width / number_cells) * 2.2), int((screen_width / number_cells) * 2.1)))

    #circle drawing
    pygame.draw.circle(screen, (245, 242, 245), ((screen_width / number_cells) * 2.5, (screen_width / number_cells) * 2.5), pix_square_size*0.4 )

    #blocks
    pygame.draw.rect(
        screen,
        (138, 73, 131),
        [(screen_width / number_cells) * 6.15, (screen_width / number_cells) * 5.15, pix_square_size*0.7, pix_square_size*0.7],
        0,
        border_radius=10,
    )

    
    pygame.display.flip()

pygame.quit()
