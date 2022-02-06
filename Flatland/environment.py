import time

from typing import List, Tuple, Union
from copy import copy
from matplotlib import pyplot as plt
import numpy as np
import pygame
from math import sin, cos, atan, atan2, pi, sqrt, hypot
import cv2
from bfs import BreadthFirstSearchPlanner
from dfs import DepthFirstSearchPlanner
from dijkstra import DijkstraPlanner
from astar import AStarPlanner

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


class Controller:
    def __init__(self, robot: Robot, sx, sy, gx, gy) -> None:
        self.robot = robot
        # self.path_x = path[:,0]
        # self.path_y = 127 - np.array(path[:,1])
        self.start_x = sx
        self.start_y = sy
        self.goal_x = gx
        self.goal_y = gy

        self.path_x = np.array([0])
        self.path_y = np.array([0])
        self.counter = 0
    
    def step(self):
        self.robot.x_coord = int(self.path_x[self.counter])
        self.robot.y_coord = int(self.path_y[self.counter])

        # print("X coord: ", self.robot.x_coord)
        # print("Y coord: ", self.robot.y_coord)
        if self.counter < len(self.path_x)-1:
            self.counter += 1


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

    def set_surface(self):
        # print(type(self.env[0,0]))
        self.surface[:,:,0] = np.array(cv2.resize(self.env*255, (self.width,self.height), interpolation=cv2.INTER_AREA),dtype=int)
        self.surface[:,:,1] = self.surface[:,:,0]
        self.surface[:,:,2] = self.surface[:,:,0]
        self.surface = 255 - self.surface

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
            self.set_surface()
        

class Telemetry:
    def __init__(self, robot: Robot, world: World) -> None:
        self.points = []
        self.robot = robot
        self.width = world.width
        self.height = world.height
        self.trace = np.zeros((self.width,self.height,3), dtype=int)
    
    def update_telemetry(self) -> None:
        self.points.append([self.robot.x_coord, self.robot.y_coord])
    
    def set_trace(self):
        env = np.zeros((128,128),dtype='float64')
        for point in self.points:
            env[point[0],point[1]] = 255
        # self.env[self.points[0,:],self.points[1,:]] = 255
        self.trace[:,:,1] = np.array(cv2.resize(env, (self.width,self.height), interpolation=cv2.INTER_AREA),dtype=int)
        self.trace[:,:,2] = self.trace[:,:,1]
        # self.trace = 255 - self.trace
        return self.trace

class Visualizer:
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    GREEN: Tuple[int, int, int] = (0, 255, 0)
    INV_RED: Tuple[int, int, int] = (0, 255, 255)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    BLUE: Tuple[int, int, int] = (0, 0, 255)

    def __init__(self, robot: Robot, controller: Controller, telemetry: Telemetry, world: World) -> None:
        pygame.init()
        pygame.font.init()
        self.world = world
        self.robot = robot
        self.telemetry = telemetry
        self.controller = controller
        self.screen = pygame.display.set_mode((world.width, world.height))
        pygame.display.set_caption('Tetromino Challenge')
        self.font = pygame.font.SysFont('freesansbolf.tff', 30)

    def display_world(self):
        # print((255 - self.world.surface))
        surf_array = self.world.surface-self.robot.robot_map()-self.telemetry.set_trace()
        surf = pygame.pixelcopy.make_surface(surf_array)
        self.screen.blit(surf, (0,0))

        center = self.world.convert_to_display(self.robot.x_coord, self.robot.y_coord)
        goal = self.world.convert_to_display(self.controller.goal_x,127-self.controller.goal_y)
        pygame.draw.circle(self.screen, self.RED, center, 10, 2)
        pygame.draw.circle(self.screen, self.GREEN, goal, 10)
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
                    self.robot.x_coord = self.controller.path_x[0]
                    self.robot.y_coord = self.controller.path_y[0]
                    self.controller.counter = 0
                    self.telemetry.points = []

        pygame.display.flip()

        return True

    def cleanup(self) -> None:
        pygame.quit()


class Runner:
    def __init__(self, robot: Robot, controller: Controller, telemetry: Telemetry, world: World, vis: Visualizer, coverage: int) -> None:
        self.robot = robot
        self.world = world
        self.vis = vis
        self.controller = controller
        self.coverage = coverage
        self.telemetry = telemetry
        

    def run(self):
        self.world.env = np.zeros((128,128))
        running = True
        generate_path = True
        try:
            np.load("env.npy")
            print("Use previous map")
            generate_new_map = False
        except:
            generate_new_map = True

        while running:

            if generate_path:
                if generate_new_map:
                    while self.world.env.sum()/(128*128) < self.coverage:
                        self.world.generate_random_tetromino()
                        print("Generating environment, percent covered: ", (self.world.env.sum()/(self.coverage*128*128))*100, end='\r')
                    np.save("env.npy", self.world.env)
                    print("\n")

                    obs = np.load("env.npy")
                    
                    obs_idx = np.argwhere(obs == 1)
                    ox = np.array(obs_idx[:,0])
                    oy = 127 - np.array(obs_idx[:,1])
                    
                    obs_map = np.flip(obs, axis=1)
                    print("Obstacle map generated")


                # generate path
                print("Generating Path")
                self.world.env = np.load("env.npy")
                obs = np.load("env.npy")
                obs_idx = np.argwhere(obs == 1)

                ox = np.array(obs_idx[:,0])
                oy = 127 - np.array(obs_idx[:,1])
                obs_map = np.flip(obs, axis=1)

                planner1 = BreadthFirstSearchPlanner(ox, oy, obs_map)
                planner2 = DepthFirstSearchPlanner(ox, oy, obs_map)
                planner3 = DijkstraPlanner(ox, oy, obs_map)
                planner4 = AStarPlanner(ox, oy, obs_map)

                rx, ry = planner3.planning(self.controller.start_x,
                                            self.controller.start_y,
                                            self.controller.goal_x,
                                            self.controller.goal_y)
                
                if np.array_equal(rx,np.array([0])):
                    self.world.env = np.zeros((128,128))
                    generate_new_map = True
                    generate_path = True
                    print("REGENERATING MAP")
                else:
                    self.controller.path_x = np.flip(rx)
                    self.controller.path_y = 127 - np.flip(ry)                    
                    
                    self.world.set_surface()              
                    generate_path = False
                    generate_new_map = False



            self.controller.step()
            self.telemetry.update_telemetry()

            running = self.vis.update_display()

            # time.sleep(0.01)
            img_name = str(self.coverage) + '.jpeg'
            pygame.image.save(self.vis.screen, img_name)


def main():
    height = int(1280*0.75)
    width = int(1280*0.75)
    sx, sy = 1, 127
    gx, gy = 126, 1
    
    robot = Robot(width, height)
    controller = Controller(robot, sx, sy, gx, gy)
    
    world = World(robot, width, height)
    telemetry = Telemetry(robot, world)
    vis = Visualizer(robot, controller, telemetry, world)

    coverage = 5
    runner = Runner(robot, controller, telemetry, world, vis, coverage/100)

    

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
