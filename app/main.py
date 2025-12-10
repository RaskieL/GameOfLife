import pygame
import numpy as np

# imports class
import particle
import world

WIDTH: int = 1280
HEIGHT: int = 720

WORLD: World = World(WIDTH, HEIGHT, 1)

def main():
    screen, clock = init()
    loop(screen, clock)

def init():    
    print("Initialisation")
    pygame.init()
    pygame.set_caption("JEU DE LA VIE :D")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    return screen, clock


def loop(screen, clock):
    while True:
        update()
        clock.tick(60)


def update():
    print("update")
    

main()