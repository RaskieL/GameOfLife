import pygame
import random
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
        
        self.grid: List[List[Particle]] = [[Particle(False, x, y) for x in range(self.cols)] for y in range(self.rows)]
        
        self.particle_img = pygame.image.load("particle.jpg").convert_alpha()
        self.particle_img = pygame.transform.scale(self.particle_img, (self.p_size, self.p_size))
        
        self.init_world()
    
    def place_star(self, star: Star, pos_x, pos_y):
        self.grid[pos_y][pos_x] = star
        for dx in range(-star.radius+1, star.radius):
            for dy in range(-star.radius+1, star.radius):
                if (dx == 0 and dy == 0) or (abs(dx) == star.radius-1 and abs(dy) == star.radius-1):
                    continue
                if 0 <= pos_y+dy < self.rows and 0 <= pos_x+dx < self.cols:
                    self.grid[pos_y+dy][pos_x+dx] = Star(0, 0, pos_x+dx, pos_y+dy)
        
    def init_world(self) -> None:
        alive_perc = 0.5
        alive = 0
        for y in range(self.rows):
            for x in range(self.cols):
                particle = self.grid[y][x]
                particle.alive = True if random.random() < alive_perc else False 
                if particle.alive:
                    alive += 1

        # Crear estrellas
        star: Star = Star(2, 4, random.randint(0, self.cols-1), random.randint(0, self.rows-1))
        star2: Star = Star(1, 3, random.randint(0, self.cols-1), random.randint(0, self.rows-1))
        star3: Star = Star(0.5, 2, random.randint(0, self.cols-1), random.randint(0, self.rows-1))
        self.stars.extend([star, star2, star3])

        for star in self.stars:
            self.place_star(star, star.pos_x, star.pos_y)

        total = self.cols * self.rows
        print(f"Alive cells: {alive}, Dead cells: {total - alive}, Percentage alive: {alive / total * 100:.2f}%")

    def handle_star_physics(self):
        for i, s1 in enumerate(self.stars):
            s1.ax = 0
            s1.ay = 0
            
            for j, s2 in enumerate(self.stars):
                if i == j: continue

                dx = s2.x - s1.x
                dy = s2.y - s1.y
                dist_sq = dx*dx + dy*dy
                dist = math.sqrt(dist_sq)

                if dist < 10: dist = 10
                if dist_sq == 0: dist_sq = 1

                force = s2.mass / dist_sq
                s1.ax += (dx / dist) * force
                s1.ay += (dy / dist) * force

            s1.vx += s1.ax
            s1.vy += s1.ay
            
            s1.x += s1.vx
            s1.y += s1.vy

            # rebotes en bordes
            if s1.x < 0 or s1.x >= self.cols:
                s1.vx *= -0.7
                s1.x = max(0, min(s1.x, self.cols - 1))
            
            if s1.y < 0 or s1.y >= self.rows:
                s1.vy *= -0.7
                s1.y = max(0, min(s1.y, self.rows - 1))

            s1.pos_x = int(s1.x)
            s1.pos_y = int(s1.y)

    def handle_particle_movement(self) -> None:
        self.handle_star_physics()
        next_grid: List[List[Particle]] = [[Particle(False, x, y) for x in range(self.cols)] for y in range(self.rows)]

        for y in range(self.rows):
            for x in range(self.cols):
                particle = self.grid[y][x]
                
                if isinstance(particle, Star) or not particle.alive:
                    continue

                particle.ax = 0
                particle.ay = 0

                for star in self.stars:
                    dx = star.x - particle.x
                    dy = star.y - particle.y
                    dist_sq = dx*dx + dy*dy
                    dist = math.sqrt(dist_sq)

                    if dist < 10: dist = 10
                    
                    force = star.mass / dist_sq
                    particle.ax += (dx / dist) * force
                    particle.ay += (dy / dist) * force

                # limitar aceleración
                particle.ax = max(min(particle.ax, 0.5), -0.5)
                particle.ay = max(min(particle.ay, 0.5), -0.5)

                particle.vx += particle.ax
                particle.vy += particle.ay
                particle.vx *= 0.98
                particle.vy *= 0.98

                particle.x += particle.vx
                particle.y += particle.vy

                nx = int(particle.x)
                ny = int(particle.y)

                # colisiones con bordes
                if nx < 0 or nx >= self.cols:
                    particle.vx *= -0.95
                    nx = max(0, min(nx, self.cols - 1))
                if ny < 0 or ny >= self.rows:
                    particle.vy *= -0.95
                    ny = max(0, min(ny, self.rows - 1))

                if next_grid[ny][nx].alive:
                    next_grid[y][x] = particle 
                else:
                    next_grid[ny][nx] = particle

        self.grid = next_grid
        
        for star in self.stars:
            self.place_star(star, star.pos_x, star.pos_y)

    def update(self) -> None:
        self.calc_alivity()
        self.handle_particle_movement()
        
    def calc_alivity(self) -> None:
        current_state = np.array([[p.alive for p in row] for row in self.grid], dtype=int)

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

        next_state = ((current_state == 1) & ((neighbors == 2) | (neighbors == 3))) | ((current_state == 0) & (neighbors == 3))

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
                    pygame.draw.circle(screen, (255,255,0),
                                       (int(particle.x)*self.p_size, int(particle.y)*self.p_size),
                                       self.p_size//2)
                    continue

                if particle.alive:
                    # usar imagen con coordenadas float
                    screen.blit(self.particle_img, (int(particle.x)*self.p_size, int(particle.y)*self.p_size))