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
        
        self.grid: List[List[Particle]] = []
        self.create_empty_world()
        
    def create_empty_world(self) -> None:
        pass
    
    def update(self) -> None:
        pass
    
    def draw(self, screen: pygame.Surface) -> None:
        pass 