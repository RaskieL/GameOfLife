from typing import Tuple

class Particle:
    
    COLOR_ALIVE: Tuple[int, int, int] = (255, 255, 255)
    COLOR_DEAD: Tuple[int, int, int] = (0, 0, 0)

    def __init__(self, posx: int, posy: int, alive: bool = False) -> None:
        self.posx: int = posx
        self.posy: int = posy
        self.alive: bool = alive