import pygame
from typing import List
from particle import Particle

class World:

    def __init__(self, p_width: int, p_height: int, p_size: int) -> None:
        self.p_width: int = p_width
        self.p_height: int = p_height
        self.p_size: int = p_size
        
        self.cols: int = p_width // p_size
        self.rows: int = p_height // p_size
        
        self.grid: List[List[Particle]] = [[Particle(True) for _ in range(self.cols)] for _ in range(self.rows)]
        self.create_empty_world()
        
    def create_empty_world(self) -> None:
        pass
    
    def update(self) -> None:
        pass
    
    def draw(self, screen: pygame.Surface) -> None:
        # Drawing the particles
        for i in range(len(self.grid)):
            row = self.grid[i]
            for j in range(len(row)):
                particle = row[j]
                if particle.alive:
                    pygame.draw.rect(screen, (255,255,255), pygame.Rect(j*self.p_size, i*self.p_size, self.p_size, self.p_size), 0)
                else:
                    pass
                    #pygame.draw(screen, (0,0,255), pygame.Rect(j*self.p_size, i*self.p_size, self.p_size, self.p_size), 1)