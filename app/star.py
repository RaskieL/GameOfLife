from particle import Particle

class Star(Particle):
    def __init__(self, mass: float, radius: int, x: float, y: float) -> None:
        super().__init__(True)
        self.mass = mass * 500
        self.radius = radius
        self.x = float(x)
        self.y = float(y)
        self.pos_x = int(x)
        self.pos_y = int(y)
        self.vx = 0.0
        self.vy = 0.0