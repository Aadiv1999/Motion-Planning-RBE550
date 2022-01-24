from operator import truediv
from pickletools import uint8
import time
from typing import List, Tuple, Union
import copy
import numpy as np
import pygame
from math import sin, cos, atan, atan2, pi, sqrt


class World:

    shape = np.array([[[0,0], [19,0], [19,-9], [29,-9], [29,9], [0,9]],
             [[0,0], [9,0], [9,39], [0,39]],
             [[0,0], [9,0], [9,9], [18,9], [18,30], [9,30], [9,18], [0,18]],
             [[0,0], [9,0], [9,29], [0,29], [0,19], [-9,19], [-9,9], [0,9]]],dtype=object)
    
    obstacle = []
    visited = []
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def generate_random_tetromino(self) -> None:
        start = np.array([np.random.randint(1,127), np.random.randint(1,127)])
        choose_shape = np.random.randint(0,4)
        # choose_shape = 3
        choose_orientation = np.random.randint(0,4)*(pi/2)
        
        rot_matrix = np.array([[cos(choose_orientation), sin(choose_orientation)],
                               [-sin(choose_orientation), cos(choose_orientation)]])
        
        
        new_obs = self.shape[choose_shape] @ rot_matrix + start*10
        
        if [[choose_shape, choose_orientation, start]] not in self.visited:
            self.obstacle.append(new_obs)
            self.visited.append([choose_shape, choose_orientation, start])
        else:
            print("Repeated pose")
        
        # self.obstacle = [new_obs]







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

        for obs in self.world.obstacle:
            # print(obs)
            pygame.draw.polygon(self.screen, self.BLACK, obs)



    def update_display(self) -> bool:
        self.screen.fill(self.WHITE)

        self.display_world()

        pygame.display.flip()

        return True

    def cleanup(self) -> None:
        pygame.quit()


class Runner:
    def __init__(self, world: World, vis: Visualizer, coverage: int) -> None:
        self.world = world
        self.vis = vis
        self.coverage = coverage

    def run(self):
        running = True
        old = 0
        new = 400

        while len(self.world.obstacle) < 2731*self.coverage:
            print(len(self.world.obstacle))
            surf_array = np.zeros((1280,1280,3), dtype=int)
            pygame.pixelcopy.surface_to_array(surf_array, self.vis.screen, kind='P', opaque=255, clear=0)
            old = np.count_nonzero(surf_array[:,:,0]==0)
            # print("Old value: ",old)
            
            self.world.generate_random_tetromino()

            self.vis.update_display()
            surf_array_new = np.zeros((1280,1280,3), dtype=int)
            pygame.pixelcopy.surface_to_array(surf_array_new, self.vis.screen, kind='P', opaque=255, clear=0)
            
            new = np.count_nonzero(surf_array_new[:,:,0]==0)
            # print("New value: ",new)

            if abs(new-old) <399:
                self.world.obstacle = self.world.obstacle[:-1]
                self.vis.update_display()

            if not running:
                pygame.quit()
            
            # time.sleep(0.02)
        
        pygame.image.save(self.vis.screen, "ten.jpeg")

def main():
    height = 1280
    width = 1280

    world = World(width, height)
    vis = Visualizer(world)

    coverage = 10
    runner = Runner(world, vis, coverage/100)

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
