from typing import Tuple
from particle import Particle

class Star(Particle):
    def __init__(self, mass: float, radius: int):
        super().__init__(True)
        self.mass = mass * 500
        self.radius = radius
        self.vx = 0
        self.vy = 0