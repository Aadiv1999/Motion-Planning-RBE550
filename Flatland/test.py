import time

from typing import List, Tuple, Union
from copy import copy
from matplotlib import path
import matplotlib.pyplot as plt
import numpy as np
import pygame
from math import sin, cos, atan, atan2, pi, sqrt, hypot
import cv2

from bfs import BreadthFirstSearchPlanner
from dfs import DepthFirstSearchPlanner
from dijkstra import DijkstraPlanner
from random_planner import RandomPlanner



class Robot:
    """
    Contains all robot parameters
    Contains controller class
    Contains robot telemetry information
    """
    x_coord: int
    y_coord: int

    def __init__(self, width, height, sx, sy, gx, gy) -> None:
        self.width =  width
        self.height = height
        self.x_coord = 0
        self.y_coord = 0
        self.map = np.zeros((128,128),dtype='float64')
        self.surface = np.zeros((self.width,self.height,3), dtype=int)
        self.controller = self.Controller(sx, sy, gx, gy)
        self.telemetry = self.Telemetry(width, height)
        
    

    def robot_map(self):
        self.map = np.zeros((128,128),dtype='float64')       
        self.map[self.x_coord, self.y_coord] = 255
        self.surface[:,:,1] = np.array(cv2.resize(self.map, (self.width,self.height), interpolation=cv2.INTER_AREA),dtype=int)
        self.surface[:,:,2] = np.array(cv2.resize(self.map, (self.width,self.height), interpolation=cv2.INTER_AREA),dtype=int)

        return self.surface


    class Controller:
        def __init__(self, sx, sy, gx, gy) -> None:
            # self.robot = robot
            # self.path_x = path[:,0]
            # self.path_y = 127 - np.array(path[:,1])
            self.start_x = sx
            self.start_y = sy
            self.goal_x = gx
            self.goal_y = gy
            self.x_coord = 0
            self.y_coord = 0

            self.path_x = np.array([0])
            self.path_y = np.array([0])
            self.counter = 0

        
        def step(self):
            self.x_coord = int(self.path_x[self.counter])
            self.y_coord = int(self.path_y[self.counter])

            if self.counter < len(self.path_x)-1:
                self.counter += 1
            else:
                self.x_coord = self.goal_x
                self.y_coord = self.goal_y

    class Telemetry:
        def __init__(self, width, height) -> None:
            self.width = width
            self.height = height
            self.points = []
            self.trace = np.zeros((self.width,self.height,3), dtype=int)
            
        
        def update_telemetry(self, x_coord: int, y_coord: int) -> None:
            self.points.append([x_coord, y_coord])
        
        def set_trace(self, color):
            env = np.zeros((128,128),dtype='float64')
            for p in self.points:
                env[p[0],p[1]] = color
            
            self.trace[:,:,1] = np.array(cv2.resize(env, (self.width,self.height), interpolation=cv2.INTER_AREA),dtype=int)
            self.trace[:,:,2] = self.trace[:,:,1]
            # self.trace = 255 - self.trace

            return self.trace


class World:
    """
    World class, defines the environment
    Used to generate random objects
    """

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
                self.env[obs[0], obs[1]] = 1

        if flag:
            self.set_surface()
        



class Visualizer:
    """
    Visualizer class is used to display the robot and world
    using Pygame
    """
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    GREEN: Tuple[int, int, int] = (0, 255, 0)
    INV_RED: Tuple[int, int, int] = (0, 255, 255)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    BLUE: Tuple[int, int, int] = (0, 0, 255)

    def __init__(self, robot: List[Robot], world: World) -> None:
        pygame.init()
        pygame.font.init()
        self.world = world
        self.robots = robot
        self.color = [self.RED, self.GREEN, self.INV_RED, self.BLUE]
        
        self.screen = pygame.display.set_mode((world.width, world.height))
        pygame.display.set_caption('Tetromino Challenge')
        self.font = pygame.font.SysFont('freesansbolf.tff', 30)

    def display_world(self):
        surf_array = self.world.surface
        for r in self.robots:
            surf_array = surf_array - r.robot_map()

        # surf_array = surf_array - self.robots[0].robot_map() - self.robots[0].telemetry.set_trace()
        
        surf = pygame.pixelcopy.make_surface(surf_array)
        self.screen.blit(surf, (0,0))

        counter = 0
        for r in self.robots:
            center = self.world.convert_to_display(r.controller.x_coord, r.controller.y_coord)
            goal = self.world.convert_to_display(r.controller.goal_x,127-r.controller.goal_y)

            pygame.draw.circle(self.screen, self.BLUE, center, 10, 2)
            pygame.draw.circle(self.screen, self.GREEN, goal, 10)
            
            path_left = (self.world.width/128)*np.array([r.controller.path_x, r.controller.path_y]).T
            
            # pygame.draw.rect(self.screen, self.BLUE, pygame.Rect(top_left[0], top_left[1], 10, 10), 2)
            for i in range(r.controller.counter):
                pygame.draw.rect(self.screen, self.color[counter], pygame.Rect(path_left[i,0], path_left[i,1], 10, 10))
            
            counter += 1


    def update_display(self, coverage) -> bool:

        self.display_world()

        for event in pygame.event.get():
            # Keypress
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # if escape is pressed, quit the program
                    return False
                if event.key == pygame.K_UP:
                    for r in self.robots:
                        r.x_coord = r.controller.path_x[0]
                        r.y_coord = r.controller.path_y[0]
                        r.controller.counter = 0
                        r.telemetry.points = []

        pygame.display.flip()
        pygame.display.set_caption('Tetromino Challenge '+str(np.round(coverage*100,0))+'%')

        return True

    def cleanup(self) -> None:
        pygame.quit()


class Runner:
    """
    Calls all the class methods to display and run the 4 planning algorithms
    """
    def __init__(self, robot: List[Robot], world: World, vis: Visualizer, coverage: int) -> None:
        self.robots = robot
        self.world = world
        self.vis = vis
        self.coverage = coverage
        # coverage | DFS | BFS | Dijkstra
        self.num_iterations = []
    
    def check_success(self):
        success = 0
        for r in self.robots:
            # print(success)
            if (r.controller.x_coord == r.controller.goal_x) and\
                            (r.controller.y_coord == r.controller.goal_y):
                success += 1
        # success == True when 2 fastest robots have reached the goal
        if success == len(self.robots) - 2:
            return True
        return False
        

    def run(self):
        
        # initialize variables

        success = False
        self.world.env = np.zeros((128,128))
        running = True
        generate_path = True
        all_paths = True
        counter = 0
        try:
            np.load("env.npy")
            print("Use previous map")
            # always generate new map
            generate_new_map = True
        except:
            generate_new_map = True

        # set a limit to maximum coverage
        while self.coverage < 0.5 and running:
            
            # generate new path
            if generate_path:
                # generate new map with different obstacle density
                if generate_new_map:
                    while self.world.env.sum()/(128*128) < self.coverage:
                        self.world.generate_random_tetromino()
                        print("Generating environment, percent completed: ", (self.world.env.sum()/(self.coverage*128*128))*100, end='\r')
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

                # initialize planners
                planner1 = BreadthFirstSearchPlanner(ox, oy, obs_map)
                planner2 = DepthFirstSearchPlanner(ox, oy, obs_map)
                planner3 = DijkstraPlanner(ox, oy, obs_map)
                planner4 = RandomPlanner(ox, oy, obs_map)

                # get the complete path using the respective planners
                rx, ry = planner2.planning(self.robots[0].controller.start_x,
                                            self.robots[0].controller.start_y,
                                            self.robots[0].controller.goal_x,
                                            self.robots[0].controller.goal_y)
                
                rx1, ry1 = planner1.planning(self.robots[1].controller.start_x,
                                            self.robots[1].controller.start_y,
                                            self.robots[1].controller.goal_x,
                                            self.robots[1].controller.goal_y)
                
                rx2, ry2 = planner3.planning(self.robots[2].controller.start_x,
                                            self.robots[2].controller.start_y,
                                            self.robots[2].controller.goal_x,
                                            self.robots[2].controller.goal_y)
                
                rx3, ry3 = planner4.planning(self.robots[3].controller.start_x,
                                            self.robots[3].controller.start_y,
                                            self.robots[3].controller.goal_x,
                                            self.robots[3].controller.goal_y)
                
                # if path does not exist, regenerate map with new obstacles
                if np.array_equal(rx,np.array([0])) or np.array_equal(rx1,np.array([0])) or np.array_equal(rx2,np.array([0])) and counter < 2:
                    self.world.env = np.zeros((128,128))
                    generate_new_map = True
                    generate_path = True
                    all_paths = False
                    print("REGENERATING MAP")                    
                    counter += 1
                    print("Tries: ", counter)
                else:
                    # else, assign path to each robot
                    self.robots[0].controller.path_x = np.flip(rx)
                    self.robots[0].controller.path_y = 127 - np.flip(ry)
                    self.robots[1].controller.path_x = np.flip(rx1)
                    self.robots[1].controller.path_y = 127 - np.flip(ry1)
                    self.robots[2].controller.path_x = np.flip(rx2)
                    self.robots[2].controller.path_y = 127 - np.flip(ry2)
                    self.robots[3].controller.path_x = np.flip(rx3)
                    self.robots[3].controller.path_y = 127 - np.flip(ry3)

                    all_paths = True               
                    
                    self.world.set_surface()              
                    generate_path = False
                    generate_new_map = False

                    self.num_iterations.append([self.coverage, len(rx1), len(rx), len(rx2), len(rx3)])


            # in every iteration of the while loop, step all robot controllers
            # update each robots telemetry information
            for r in self.robots:
                r.controller.step()
                r.telemetry.update_telemetry(r.controller.x_coord, r.controller.y_coord)

            # check if the fastest 2 robots have reached the goal
            success = self.check_success()

            # if successful and all planners produce a path
            # increment the obstacle density and regenerate map
            if success and all_paths:
                self.coverage += 0.05
                print("NEW COVERAGE = ", self.coverage*100)
                self.world.env = np.zeros((128,128))
                for r in self.robots:
                    r.controller.path_x = np.array([0])
                    r.controller.path_y = np.array([0])
                    r.controller.counter = 0
                    r.telemetry.points = []
                    r.telemetry.set_trace(255)
                generate_path = True
                generate_new_map = True
            
            # after every iteration of the while loop, update the display
            running = self.vis.update_display(self.coverage)

            # time.sleep(0.01)
            
            # save the pygame surface as an image
            self.coverage = np.round(self.coverage, 2)
            img_name = str(self.coverage) + '.jpeg'
            pygame.image.save(self.vis.screen, img_name)
        
        np.save('plot_data', self.num_iterations)


def main():
    height = int(1280*0.75)
    width = int(1280*0.75)
    sx, sy = 1, 127
    gx, gy = 126, 0
    
    # instantiate all robots
    robot = Robot(width, height, sx, sy, gx, gy)
    robot1 = Robot(width, height, sx, sy, gx, gy)
    robot2 = Robot(width, height, sx, sy, gx, gy)
    robot3 = Robot(width, height, sx, sy, gx, gy)

    robots = [robot, robot1, robot2, robot3]
       
    world = World(robots, width, height)
    
    vis = Visualizer(robots, world)

    # seed coverage %
    coverage = 5
    runner = Runner(robots, world, vis, coverage/100)

    

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
