class Particle:
    
    BLACK = (0, 0, 0)
    GRAY = (50, 50, 50)
    WHITE = (255, 255, 255)
    
    pos_x: int
    pos_y: int

    value: bool
    

    def __init__(self, posx, posy, val):
        pos_x = posx
        pos_y = posy
        value = val