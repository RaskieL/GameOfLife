import pygame
from random import random as rnd
from typing import List
from particle import Particle
from star import Star
import numpy as np

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

        star: Star = Star(50, 3)
        middle_x = self.cols//2
        middle_y = self.rows//2
        self.grid[middle_y][middle_x] = star
        for dx in range(-star.radius+1,star.radius):
            for dy in range(-star.radius+1,star.radius):
                self.grid[middle_y+dy][middle_x+dx] = Star(0,0)

        total = self.cols * self.rows
        print(f"Alive cells: {alive}, Dead cells: {total - alive}, Percentage alive: {alive / total * 100}%")
    
    def update(self) -> None:
        self.calc_alivity()


    def calc_alivity(self) -> None:
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
    
    def draw(self, screen: pygame.Surface) -> None:
        # Drawing the particles
        for y in range(len(self.grid)):
            row = self.grid[y]
            for x in range(len(row)):
                particle = row[x]
                if isinstance(particle, Star):
                    pygame.draw.rect(screen, (255,255,0), pygame.Rect(x*self.p_size, y*self.p_size, self.p_size, self.p_size), 0)
                    continue

                if particle.alive:
                    pygame.draw.rect(screen, (255,255,255), pygame.Rect(x*self.p_size, y*self.p_size, self.p_size, self.p_size), 0)
                else:
                    continue