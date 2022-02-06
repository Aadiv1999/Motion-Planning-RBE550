import numpy as np
import math



class RandomPlanner:

    def __init__(self, ox, oy, obs_map):

        self.reso = 1
        self.robot_radius = 0.5
        self.minx = 0
        self.miny = 0
        self.maxx = 127
        self.maxy = 127

        self.x_width = round((self.maxx - self.minx) / self.reso)
        self.y_width = round((self.maxy - self.miny) / self.reso)

        self.obstacle_map = obs_map
    
    def get_next_grid(self, curr_x, curr_y):
        next_grid = np.random.randint(0,8)
        if next_grid == 0:
            curr_x += 1
            curr_y += 0

        if next_grid == 1:
            curr_x += 0
            curr_y += 1

        if next_grid == 2:
            curr_x += -1
            curr_y += 0

        if next_grid == 3:
            curr_x += 0
            curr_y += -1

        if next_grid == 4:
            curr_x += 1
            curr_y += 1

        if next_grid == 5:
            curr_x += -1
            curr_y += 1

        if next_grid == 6:
            curr_x += -1
            curr_y += -1

        if next_grid == 7:
            curr_x += 1
            curr_y += -1
        
        return curr_x, curr_y


    def planning(self, sx, sy, gx, gy):
        curr_x = sx
        curr_y = sy
        rx = []
        ry = []
        iterations = 0

        while iterations < 1000:
            new_x, new_y = self.get_next_grid(curr_x, curr_y)

            if curr_x < self.minx or curr_x > self.maxx or curr_y < self.miny or curr_y > self.maxy or \
                    self.obstacle_map[curr_x, 127-curr_y] == 1:
                iterations += 1
                rx.append(curr_x)
                ry.append(curr_y)                
            else:
                rx.append(new_x)
                ry.append(new_y)
                curr_x = new_x
                curr_y = new_y
                iterations += 1
        return rx, ry

