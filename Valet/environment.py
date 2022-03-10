from mouse import mouse_trajectory
import time
from typing import List, Tuple, Union
from copy import copy
import numpy as np
import pygame
import random
from math import sin, cos, atan, atan2, pi, sqrt


def max_speed(speed):
    if speed > 1:
        return 1
    else:
        return speed


def rot_points(mat, degrees: float):
    rot = []
    degrees = degrees * pi/180
    rot_mat = np.array([[cos(degrees), sin(degrees)],[-sin(degrees), cos(degrees)]])

    return mat @ rot_mat

def convert_to_display(mat):
    mat[:,1] = 1000 - mat[:,1]
    return mat

class Robot:
    x_coord: int
    y_coord: int
    vel_r: int
    vel_l: int
    width: int = 20
    height: int = 30
    angle = 0
    points = []
    dt = 0 


    def __init__(self, x, y, theta) -> None:
        self.x_coord = x
        self.y_coord = y
        self.angle = theta
        self.vel_l = 0
        self.vel_r = 0
        
        # self.chassis = self.robot_points()
    
    def get_robot_points(self):
        points = []
        
        points.append([self.width+20,0])
        points.append([self.width/2, self.height/2])
        points.append([-self.width/2, self.height/2])
        points.append([-self.width/2, -self.height/2])
        points.append([self.width/2, -self.height/2])
        

        # right wheel
        points.append([0, self.height/2])
        # left wheel
        points.append([0, -self.height/2])

        
        points = rot_points(points, self.angle) + np.array([self.x_coord, self.y_coord])
        points = convert_to_display(points)
        
        return points

    def move(self, vel_l, vel_r) -> None:
        self.vel_r = vel_r
        self.vel_l = vel_l
    
    def get_position(self) -> Tuple[float, float]:
        return self.x_coord, self.y_coord
    
    def get_speed(self) -> Tuple[float, float]:
        return self.vel_l, self.vel_r

    def set_position(self, goal: Tuple[float, float]) -> None:
        self.x_coord = goal[0]
        self.y_coord = goal[1]

        

class Controller:

    def __init__(self, robot: Robot) -> None:
        self.robot = robot

    def step(self):
        # theta = pi - self.robot.angle * pi/180
        theta = self.robot.angle * pi/180
        self.robot.x_coord += ((self.robot.vel_l+self.robot.vel_r)/2)*cos(theta) * self.robot.dt
        self.robot.y_coord += ((self.robot.vel_l+self.robot.vel_r)/2)*sin(theta) * self.robot.dt
        self.robot.angle += atan2((self.robot.vel_r - self.robot.vel_l),self.robot.height)*180/pi * self.robot.dt

        self.robot.angle = self.robot.angle % 360
    
    def check_success(self, goal: Tuple[float, float]) -> bool:
        return np.allclose(self.robot.get_position(), goal, atol=3.5)
    

class World:    
    obstacle = []
    obs = np.array([[200,100],
                    [-200,100],
                    [-200,-100],
                    [200,-100],
                    [200,100]])
    vehicle1 = []
    vehicle2 = []

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.obs = self.obs + np.array([600,400])
        
        
class Planner:

    
    def __init__(self) -> None:
        self.trajectory = [[500, 800],
                            [900,800],
                            [800, 500],
                            [20,50],
                            [900, 900],
                            [500,500],
                            [100, 100]]

    def get_trajectory(self):
        return self.trajectory

    def get_heading(self, dx, dy) -> float:
        heading = atan2(dy,dx) * 180/pi
        # print("head: ", heading)
        if heading < 0:
            return 360 + heading
        else:
            return heading

class Visualizer:
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    BLUE: Tuple[int, int, int] = (0, 0, 255)
    TREE_GREEN: Tuple[int, int, int] = (34, 139, 34)
    CROP_GREEN: Tuple[int, int, int] = (167, 199, 155)
    WEED_GREEN: Tuple[int, int, int] = (20, 66, 6)

    def __init__(self, robot: Robot, world: World, planner: Planner) -> None:
        pygame.init()
        pygame.font.init()
        self.robot = robot
        self.world = world
        self.planner = planner
        self.screen = pygame.display.set_mode((world.width, world.height))
        pygame.display.set_caption('Farmland')
        self.font = pygame.font.SysFont('freesansbolf.tff', 30)
    
    def display_robot(self):
        robot_points = self.robot.get_robot_points()
        pygame.draw.circle(self.screen, self.BLACK, (self.robot.x_coord, 1000-self.robot.y_coord),5)
        for i in range(5):
            if i == 4:
                pygame.draw.line(self.screen, self.RED, robot_points[i], robot_points[0], 4)
            else:
                pygame.draw.line(self.screen, self.RED, robot_points[i], robot_points[i+1], 4)
        
        # right wheel
        pygame.draw.circle(self.screen, self.BLUE, robot_points[5], 5)
        # left wheel
        pygame.draw.circle(self.screen, self.BLACK, robot_points[6], 5)

        coor = 'Coordinates: ' + str(np.round(self.robot.get_position(),2))
        speeds = 'Speeds: ' + str(np.round(self.robot.get_speed(),2))
        text = self.font.render(coor, True, self.BLACK)
        self.screen.blit(text, (1, 30))
        text1 = self.font.render(speeds, True, self.BLACK)
        self.screen.blit(text1, (1, 5))


    def display_world(self):
        # pygame.draw.circle(self.screen, self.BLACK, (self.world.width/2, self.world.height/2),5)
        for i in range(len(self.world.obs)-1):
            pygame.draw.line(self.screen, self.RED, self.world.obs[i], self.world.obs[i+1], 8)
        
        traj = convert_to_display(np.array(self.planner.get_trajectory()))

        for i in range(len(traj)-1):
            pygame.draw.line(self.screen, self.BLUE, traj[i], traj[i+1])

    def update_display(self) -> bool:

        self.screen.fill(self.WHITE)

        self.display_world()

        self.display_robot()

        for event in pygame.event.get():
            # Keypress
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # if escape is pressed, quit the program
                    return False

        pygame.display.flip()

        return True

    def cleanup(self) -> None:
        pygame.quit()


class Runner:
    def __init__(self, robot: Robot, controller: Controller, world: World, planner: Planner, vis: Visualizer) -> None:
        self.robot = robot
        self.controller = controller
        self.world = world
        self.planner = planner
        self.vis = vis      
        

    def run(self, trajectory):
        self.planner.trajectory = convert_to_display( np.array(trajectory))
        x = self.robot.x_coord
        y = self.robot.y_coord
        running = True
        counter = 0
        last_counter = -1
        lasttime = pygame.time.get_ticks() 
        success = False     
        goal = self.planner.get_trajectory()

        while running:
            
            self.controller.step()

            if counter < len(goal):
                success = self.controller.check_success(goal[counter])
            else:
                success = True
                print("Final goal reached")
            
            if success:
                # do stuff for new goal
                # stop robot
                self.robot.move(0,0)
                # if counter < len(goal):
                #     self.robot.set_position(goal[counter])
                counter += 1
                
                print("Waypoint reached")
                print(self.robot.get_position())
            else:
                # turn towards goal
                if counter == 0:
                    dy = goal[counter][1] - y
                    dx = goal[counter][0] - x
                    last_counter = counter
                else:
                    # dy = goal[counter][1] - goal[counter-1][1]
                    # dx = goal[counter][0] - goal[counter-1][0]
                    if counter != last_counter:
                        dy = goal[counter][1] - self.robot.y_coord
                        dx = goal[counter][0] - self.robot.x_coord
                        last_counter = counter
                
                heading = np.round(self.planner.get_heading(dx,dy),0)
                diff_heading = self.robot.angle - heading
                
                if abs(diff_heading) > 0.05:
                    turn_speed = 0.01
                    if diff_heading >= 0:
                        if diff_heading <=180:
                            self.robot.move(turn_speed, -turn_speed)
                        else:
                            self.robot.move(-turn_speed, turn_speed)
                    else:
                        if diff_heading <= -180:
                            self.robot.move(turn_speed, -turn_speed)
                        else:
                            self.robot.move(-turn_speed, turn_speed)
                else:
                    # self.robot.angle = heading
                    gain = 0.005
                    speed = max_speed(gain*((self.robot.x_coord-goal[counter][0])**2 + (self.robot.y_coord-goal[counter][1])**2)**0.5)
                    self.robot.move(speed, speed)
                



            

            # dt            
            self.robot.dt = (pygame.time.get_ticks() - lasttime)
            lasttime = pygame.time.get_ticks()

            running = self.vis.update_display()
            
            time.sleep(0.001)
        

def main():
    height = 1000
    width = 1000


    robot = Robot(0,900,0)
    controller = Controller(robot)
    world = World(width, height)
    planner = Planner()
    vis = Visualizer(robot, world, planner)

    runner = Runner(robot, controller, world, planner, vis)

    try:
        trajectory = mouse_trajectory(world.obs, width, height)
        print(len(trajectory))
        traj = trajectory[1::3]
        print(len(traj))
        runner.run(traj)
    except AssertionError as e:
        print(f'ERROR: {e}, Aborting.')
    except KeyboardInterrupt:
        pass
    finally:
        vis.cleanup()


if __name__ == '__main__':
    main()
