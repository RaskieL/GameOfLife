from typing import Tuple
from particle import Particle

class Star(Particle):
    mass: float
    radius: int

    def __init__(self, mass: float, radius: int) -> None:
        self.alive: bool = True
        self.mass = mass
        self.radius = radius