from typing import Tuple

class Particle:
    
    COLOR_ALIVE: Tuple[int, int, int] = (255, 255, 255)
    COLOR_DEAD: Tuple[int, int, int] = (0, 0, 0)

    def __init__(self, alive: bool = False) -> None:
        self.alive: bool = alive