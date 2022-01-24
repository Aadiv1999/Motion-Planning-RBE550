from operator import truediv
import time
from typing import List, Tuple, Union
import copy
import numpy as np
import pygame
# import matplotlib.path as mplPath
# import matplotlib.pylab as plt
# from math import sin, cos, atan, atan2, pi, sqrt


class World:
    obstacles = [[(10,10), (30,10), (30,0), (40,0), (40,20), (10,20)]]
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def generate_random_tetromino(self):
        pass




class Visualizer:
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    BLUE: Tuple[int, int, int] = (0, 0, 255)

    def __init__(self, world: World) -> None:
        pygame.init()
        pygame.font.init()
        self.world = world
        self.screen = pygame.display.set_mode((world.width, world.height))
        pygame.display.set_caption('Tetromino Challenge')
        self.font = pygame.font.SysFont('freesansbolf.tff', 30)

    def display_world(self):

        for obs in self.world.obstacles:
            pygame.draw.polygon(self.screen, self.BLACK, obs)



    def update_display(self) -> bool:
        self.screen.fill(self.WHITE)

        self.display_world()

        pygame.display.flip()

        return True

    def cleanup(self) -> None:
        pygame.quit()


class Runner:
    def __init__(self, world: World, vis: Visualizer) -> None:
        self.world = world
        self.vis = vis

    def run(self):
        running = True

        while running:

            self.world.generate_random_tetromino()

            self.vis.update_display()

            if not running:
                pygame.quit()
            
            time.sleep(0.01)

def main():
    height = 1280
    width = 1280

    world = World(width, height)
    vis = Visualizer(world)

    runner = Runner(world, vis)

    try:
        runner.run()
    except AssertionError as e:
        print(f'ERROR: {e}, Aborting.')
    except KeyboardInterrupt:
        pass
    finally:
        vis.cleanup()


if __name__ == '__main__':
    main()