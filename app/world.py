import pygame
from random import random as rnd
from typing import List
from particle import Particle
from star import Star
import numpy as np
import math

class World:
    stars: List[Star] = []

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
    
    def place_star(self, star: Star, pos_x, pos_y):
        self.grid[pos_y][pos_x] = star
        for dx in range(-star.radius+1,star.radius):
            for dy in range(-star.radius+1,star.radius):
                if (dx == 0 and dy == 0) or (abs(dx) == 2 and abs(dy) == 2):
                    continue
                self.grid[pos_y+dy][pos_x+dx] = Star(0,0)
        
    def init_world(self) -> None:
        alive_perc = 0.2
        alive = 0
        for y in range(len(self.grid)):
            row = self.grid[y]
            for x in range(len(row)):
                particle = row[x]
                particle.alive = True if rnd() < alive_perc else False 
                if particle.alive:
                    alive += 1

        star: Star = Star(1, 3)
        self.stars.append(star)

        # placing star
        middle_x = self.cols//2
        middle_y = self.rows//2

        for star in self.stars:
            self.place_star(star,middle_x,middle_y)

        total = self.cols * self.rows
        print(f"Alive cells: {alive}, Dead cells: {total - alive}, Percentage alive: {alive / total * 100}%")

    def handle_particle_movement(self) -> None:
        middle_x = self.cols // 2
        middle_y = self.rows // 2
        
        next_grid: List[List[Particle]] = [[Particle(False) for _ in range(self.cols)] for _ in range(self.rows)]

        for y in range(self.rows):
            for x in range(self.cols):
                particle = self.grid[y][x]
                
                if isinstance(particle, Star):
                    continue
                
                if not particle.alive:
                    continue

                particle.ax = 0
                particle.ay = 0

                for star in self.stars:
                    dx = middle_x - x
                    dy = middle_y - y
                    dist_sq = dx*dx + dy*dy
                    dist = math.sqrt(dist_sq)

                    if dist < 5: dist = 5
                    
                    force = star.mass / (dist_sq * 1)
                    
                    # Vecteurs normalisés * force
                    particle.ax += (dx / dist) * force
                    particle.ay += (dy / dist) * force

                # 3. Application Mouvement + Frottements
                particle.vx += particle.ax
                particle.vy += particle.ay
                
                particle.vx *= 0.95
                particle.vy *= 0.95

                nx = int(x + particle.vx)
                ny = int(y + particle.vy)

                # Collisions murs
                if nx < 0 or nx >= self.cols:
                    particle.vx *= -1 # Rebond
                    nx = max(0, min(nx, self.cols - 1))
                
                if ny < 0 or ny >= self.rows:
                    particle.vy *= -1 # Rebond
                    ny = max(0, min(ny, self.rows - 1))

                # Gestion des collisions entre particules
                # Si la case cible est déjà occupée par une particule vivante dans next_grid
                if next_grid[ny][nx].alive:
                    next_grid[y][x] = particle 
                else:
                    next_grid[ny][nx] = particle

        for star in self.stars:
            self.place_star(star, middle_x, middle_y)
            next_grid[middle_y][middle_x] = star

        self.grid = next_grid
        
    
    def update(self) -> None:
        self.calc_alivity()
        self.handle_particle_movement()
        

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

        # Détermination du futur état (logique booléenne)
        next_state = ((current_state == 1) & ((neighbors == 2) | (neighbors == 3))) | ((current_state == 0) & (neighbors == 3))

        # Réinjection
        for y in range(self.rows):
            for x in range(self.cols):
                new_alive = bool(next_state[y, x])
                if self.grid[y][x].alive != new_alive:
                    self.grid[y][x].alive = new_alive
                
    def draw(self, screen: pygame.Surface) -> None:
        for y in range(self.rows):
            for x in range(self.cols):
                particle = self.grid[y][x]
                if isinstance(particle, Star):
                    pygame.draw.rect(screen, (255,255,0), pygame.Rect(x*self.p_size, y*self.p_size, self.p_size, self.p_size), 0)
                    continue

                if particle.alive:
                    pygame.draw.rect(screen, (255,255,255), pygame.Rect(x*self.p_size, y*self.p_size, self.p_size, self.p_size), 0)
                    #screen.blit(self.particle_img, (x*self.p_size, y*self.p_size))
                else:
                    continue
