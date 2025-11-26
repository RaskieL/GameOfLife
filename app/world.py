import particle

class World:
    WIDTH: int
    HEIGHT: int
    
    SIZE: int
    
    COLS: int
    ROWS: int

    grid: Particle[int][int]

    def __init__(self, p_width: int, p_height: int, p_size: int):
        WIDTH = p_width
        HEIGHT = p_height
        
        SIZE = p_size

        ROWS = HEIGHT // SIZE
        COLS = WIDTH  // SIZE

        grid: int = [COLS][ROWS]
        
    def create_empty_world():
        pass