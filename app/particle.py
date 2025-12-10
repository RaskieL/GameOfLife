from typing import Tuple
from random import random as rnd

class Particle:
    __slots__ = ['alive', 'ax', 'ay', 'vx', 'vy']

    def __init__(self, alive: bool = False):
        self.alive = alive
        self.ax = 0.0
        self.ay = 0.0
        self.vx = (rnd() - 0.5) * 0.5 
        self.vy = (rnd() - 0.5) * 0.5