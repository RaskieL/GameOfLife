import pygame
import numpy as np
import sys
from typing import Tuple

# imports class
from world import World

WIDTH: int = 1920
HEIGHT: int = 1080
SIZE : int = 3

def init() -> Tuple[pygame.Surface, pygame.time.Clock]:    
    print("Initialisation...")
    pygame.init()
    pygame.display.set_caption("JEU DE LA VIE ;-D")
    
    screen: pygame.Surface = pygame.display.set_mode((WIDTH, HEIGHT))
    clock: pygame.time.Clock = pygame.time.Clock()
    
    return screen, clock

def main() -> None:
    screen, clock = init()
    
    world_simulation: World = World(WIDTH, HEIGHT, SIZE)
    
    running: bool = True
    paused: bool = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_q:
                    running = False
                    

        if not paused:
            world_simulation.update()
        
        screen.fill((0, 0, 0))
        world_simulation.draw(screen)


        
        pygame.display.flip()
        
        clock.tick(60)

    pygame.quit()
    sys.exit()
    

if __name__ == "__main__":
    main()