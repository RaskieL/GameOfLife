from particle import Particle

class Star(Particle):
    mass: float
    radius: int
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float

    def __init__(self, mass: float, radius: int, x: float, y: float) -> None:
        super().__init__(True, int(x), int(y))
        
        self.mass = mass * 500
        self.radius = radius
        
        self.x = float(x)
        self.y = float(y)
        
        self.pos_x = int(x)
        self.pos_y = int(y)
        
        self.vx = 0.0
        self.vy = 0.0

        self.ax = 0.0
        self.ay = 0.0