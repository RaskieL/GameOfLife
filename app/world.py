import pygame
from random import random as rnd
from typing import List
from particle import Particle

class World:

    def __init__(self, p_width: int, p_height: int, p_size: int) -> None:
        self.p_width: int = p_width
        self.p_height: int = p_height
        self.p_size: int = p_size
        
        self.cols: int = p_width // p_size
        self.rows: int = p_height // p_size
        
        self.grid: List[List[Particle]] = [[Particle(False) for _ in range(self.cols)] for _ in range(self.rows)]
        self.init_world()
        
    def init_world(self) -> None:
        alive_perc = 0.1
        alive = 0
        for y in range(len(self.grid)):
            row = self.grid[y]
            for x in range(len(row)):
                particle = row[x]
                particle.alive = True if rnd() < alive_perc else False 
                if particle.alive:
                    alive += 1
        total = self.cols * self.rows
        print(f"Alive cells: {alive}, Dead cells: {total - alive}, Percentage alive: {alive / total * 100}%")
    
    def update(self) -> None:
        for y in range(len(self.grid)):
            row = self.grid[y]
            for x in range(len(row)):
                particle = row[x]
                
                alive_neighbours: int = 0

                for kx in range(-1, 1):
                    for ky in range(-1, 1):
                        if(kx == 0 and ky == 0):
                            continue
                        nx = x + kx
                        ny = y +ky
                        if nx >= 0 and nx < self.p_size and ny >= 0 and ny < self.p_size and self.grid[ny][nx].alive:
                            alive_neighbours += 1
                match (particle.alive):
                    case True:
                        if alive_neighbours != 3 or alive_neighbours != 2:
                            particle.alive = False
                    case False:
                        if alive_neighbours == 3:
                            particle.alive = True
                # end match
    
    def draw(self, screen: pygame.Surface) -> None:
        # Drawing the particles
        for y in range(len(self.grid)):
            row = self.grid[y]
            for x in range(len(row)):
                particle = row[x]
                if particle.alive:
                    pygame.draw.rect(screen, (255,255,255), pygame.Rect(x*self.p_size, y*self.p_size, self.p_size, self.p_size), 0)
                else:
                    continue