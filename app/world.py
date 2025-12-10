import pygame
from random import random as rnd
from typing import List
from particle import Particle
import numpy as np

class World:
    def __init__(self, p_width: int, p_height: int, p_size: int) -> None:
        self.p_width = p_width
        self.p_height = p_height
        self.p_size = p_size
        
        self.cols = p_width // p_size
        self.rows = p_height // p_size
        
        self.grid: List[List[Particle]] = [[Particle(False) for _ in range(self.cols)] for _ in range(self.rows)]
        
        self.particle_img = pygame.image.load("particle.jpg").convert_alpha()
        self.particle_img = pygame.transform.scale(self.particle_img, (self.p_size, self.p_size))
        
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
        current_state = np.array([[p.alive for p in row] for row in self.grid], dtype=int)

        # Calcul vectoriel des voisins
        neighbors = (
            np.roll(current_state, 1, axis=0) +
            np.roll(current_state, -1, axis=0) +
            np.roll(current_state, 1, axis=1) +
            np.roll(current_state, -1, axis=1) +
            np.roll(np.roll(current_state, 1, axis=0), 1, axis=1) +
            np.roll(np.roll(current_state, 1, axis=0), -1, axis=1) +
            np.roll(np.roll(current_state, -1, axis=0), 1, axis=1) +
            np.roll(np.roll(current_state, -1, axis=0), -1, axis=1)
        )

        # Détermination du futur état (Logique booléenne)
        next_state = ((current_state == 1) & ((neighbors == 2) | (neighbors == 3))) | ((current_state == 0) & (neighbors == 3))

        # Réinjection
        for y in range(self.rows):
            for x in range(self.cols):
                new_alive = bool(next_state[y, x])
                if self.grid[y][x].alive != new_alive:
                    self.grid[y][x].alive = new_alive
    
    # def draw(self, screen: pygame.Surface) -> None:
    #     # Drawing the particles
    #     for y in range(len(self.grid)):
    #         row = self.grid[y]
    #         for x in range(len(row)):
    #             particle = row[x]
    #             if particle.alive:
    #                 rect = pygame.Rect(x*self.p_size, y*self.p_size, self.p_size, self.p_size)
    #                 pygame.draw.rect(screen, (255,255,255), rect)
    #                 pygame.draw.rect(screen, (0, 255, 0), rect, 1) 
    #             else:
    #                 continue
                
    def draw(self, screen: pygame.Surface) -> None:
        for y in range(self.rows):
            for x in range(self.cols):
                particle = self.grid[y][x]
                if particle.alive:
                    screen.blit(self.particle_img, (x*self.p_size, y*self.p_size))
