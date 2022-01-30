from operator import truediv
from pickletools import uint8
import time
from tkinter import Toplevel
from turtle import width, window_width
from typing import List, Tuple, Union
from copy import copy
import numpy as np
import pygame
from math import sin, cos, atan, atan2, pi, sqrt
import cv2

class Robot:
    x_coord: int
    y_coord: int

    def __init__(self, width, height) -> None:
        self.width =  width
        self.height = height
        self.x_coord = 0
        self.y_coord = 0
        self.map = np.zeros((128,128),dtype='float64')
        self.surface = np.zeros((self.width,self.height,3), dtype=int)
    

    def robot_map(self):
        self.map = np.zeros((128,128),dtype='float64')       
        self.map[self.x_coord, self.y_coord] = 255
        self.surface[:,:,1] = np.array(cv2.resize(self.map, (self.width,self.height), interpolation=cv2.INTER_AREA),dtype=int)
        self.surface[:,:,2] = np.array(cv2.resize(self.map, (self.width,self.height), interpolation=cv2.INTER_AREA),dtype=int)

        return self.surface


class World:

    shape = np.array([[[0,0], [1,0], [2,0], [2,1]],
            [[0,0], [1,0], [2,0], [3,0]],
            [[0,0], [0,1], [0,2], [1,1]],
            [[0,0], [0,1], [1,1], [1,2]]],dtype=object)
    

    obstacle = []
    visited = []

    def __init__(self, robot: Robot, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.robot = robot
        self.env = np.zeros((128,128),dtype=int)
        self.surface = np.zeros((self.width,self.height,3), dtype=int)

    def convert_to_display(self, x_coord, y_coord):
        return (self.width/128)*np.array([x_coord, y_coord]) + [5,5]


    def generate_random_tetromino(self) -> None:
        possible_starts = np.argwhere(self.env == 0)
        start = possible_starts[np.random.randint(0,len(possible_starts))]

        choose_shape = np.random.randint(0,4)
        # choose_shape = 2
        choose_orientation = np.random.randint(0,2)*(pi/2)
        # choose_orientation = 2 * pi/2
        
        rot_matrix = np.array([[cos(choose_orientation), sin(choose_orientation)],
                               [-sin(choose_orientation), cos(choose_orientation)]])
        
        
        new_obs = np.array(self.shape[choose_shape] @ rot_matrix + start, dtype=int)
        
        original_env = copy(self.env)

        greater = new_obs > 127
        less = new_obs < 0

        if greater.sum()>0 or less.sum()>0:
            # print("Out of bounds")
            flag = 0
            self.env = original_env
        else:
            for obs in new_obs:
                flag = 1
                if self.env[obs[0], obs[1]] == 1 or (obs[1]==0 and obs[0]==0) or (obs[1]==127 and obs[0]==127):
                    self.env = original_env
                    flag = 0
                    break
                else:
                    self.env[obs[0], obs[1]] = 1

        if flag:
            # print(type(self.env[0,0]))
            self.surface[:,:,0] = np.array(cv2.resize(self.env*255, (self.width,self.height), interpolation=cv2.INTER_AREA),dtype=int)
            self.surface[:,:,1] = self.surface[:,:,0]
            self.surface[:,:,2] = self.surface[:,:,0]
            self.surface = 255 - self.surface
        


class Visualizer:
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    INV_RED: Tuple[int, int, int] = (0, 255, 255)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    BLUE: Tuple[int, int, int] = (0, 0, 255)

    def __init__(self, robot: Robot, world: World) -> None:
        pygame.init()
        pygame.font.init()
        self.world = world
        self.robot = robot
        self.screen = pygame.display.set_mode((world.width, world.height))
        pygame.display.set_caption('Tetromino Challenge')
        self.font = pygame.font.SysFont('freesansbolf.tff', 30)

    def display_world(self):
        # print((255 - self.world.surface))
        surf = pygame.pixelcopy.make_surface(self.world.surface-self.robot.robot_map())
        self.screen.blit(surf, (0,0))
        center = self.world.convert_to_display(self.robot.x_coord, self.robot.y_coord)
        pygame.draw.circle(self.screen, self.BLUE, center, 10, 2)
        top_left = (self.world.width/128)*np.array([self.robot.x_coord, self.robot.y_coord])
        pygame.draw.rect(self.screen, self.BLUE, pygame.Rect(top_left[0], top_left[1], 10, 10), 2)


    def update_display(self) -> bool:

        self.display_world()

        for event in pygame.event.get():
            # Keypress
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # if escape is pressed, quit the program
                    return False
                if event.key == pygame.K_UP:
                    self.robot.x_coord = 0
                    self.robot.y_coord = 0

        pygame.display.flip()

        return True

    def cleanup(self) -> None:
        pygame.quit()


class Runner:
    def __init__(self, robot: Robot, world: World, vis: Visualizer, coverage: int) -> None:
        self.robot = robot
        self.world = world
        self.vis = vis
        self.coverage = coverage
        

    def run(self):
        self.world.env = np.zeros((128,128))
        running = True

        while running:

            while self.world.env.sum()/(128*128) < self.coverage:
                self.world.generate_random_tetromino()
                print("Generating environment, percent covered: ", (self.world.env.sum()/(self.coverage*128*128))*100, end='\r')

            np.save("env.npy", self.world.env)
            
            if self.robot.x_coord < 127:
                self.robot.x_coord += 1
                self.robot.y_coord += 1

            running = self.vis.update_display()

            time.sleep(0.1)
        

        img_name = str(self.coverage) + '.jpeg'
        pygame.image.save(self.vis.screen, img_name)

def main():
    height = int(1280*0.75)
    width = int(1280*0.75)
    
    robot = Robot(width, height)
    world = World(robot, width, height)
    vis = Visualizer(robot, world)

    coverage = 30
    runner = Runner(robot, world, vis, coverage/100)

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
