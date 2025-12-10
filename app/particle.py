from random import random as rnd

class Particle:
    __slots__ = ['alive', 'ax', 'ay', 'vx', 'vy', 'x', 'y']

    def __init__(self, alive: bool = False, x: float = 0.0, y: float = 0.0):
        self.alive = alive
        self.ax = 0.0
        self.ay = 0.0
        
        self.vx = (rnd() - 0.5) * 0.5
        self.vy = (rnd() - 0.5) * 0.5

        self.x = float(x)
        self.y = float(y)