from random import random as rnd
from typing import Union

class Particle:
    __slots__ = ['alive', 'ax', 'ay', 'vx', 'vy', 'x', 'y']

    def __init__(self, alive: bool = False, x: Union[int, float] = 0.0, y: Union[int, float] = 0.0) -> None:
        self.alive: bool = alive
        
        self.ax: float = 0.0
        self.ay: float = 0.0
        
        self.vx: float = (rnd() - 0.5) * 0.5
        self.vy: float = (rnd() - 0.5) * 0.5

        self.x: float = float(x)
        self.y: float = float(y)