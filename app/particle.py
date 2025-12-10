from typing import Tuple

class Particle:
    alive: bool

    def __init__(self, alive: bool = False) -> None:
        self.alive: bool = alive