"""

Grid based Dijkstra planning

author: Atsushi Sakai(@Atsushi_twi)

"""

import numpy as np
import math



class DijkstraPlanner:

    def __init__(self, ox, oy, obs_map):
        """
        Initialize map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        reso: grid reso [m]
        rr: robot radius[m]
        """
        self.reso = 1
        self.robot_radius = 0.5
        self.motion = self.get_motion_model()
        self.minx = 0
        self.miny = 0
        self.maxx = 127
        self.maxy = 127

        self.x_width = round((self.maxx - self.minx) / self.reso)
        self.y_width = round((self.maxy - self.miny) / self.reso)

        self.obstacle_map = obs_map

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index  # index of previous Node

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.minx),
                               self.calc_xy_index(sy, self.miny), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.minx),
                              self.calc_xy_index(gy, self.miny), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("DIJKSTRA Open set is empty..")
                return np.array([0]), np.array([127])
            
            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]
            # print(c_id)

            if current.x == goal_node.x and current.y == goal_node.y:
                # print("Found goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand search grid based on motion model
            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_index(node)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # Discover a new node
                else:
                    if open_set[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_position(goal_node.x, self.minx)], [
            self.calc_position(goal_node.y, self.miny)]
        parent_index = goal_node.parent_index

        while parent_index != -1:
            
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.minx))
            ry.append(self.calc_position(n.y, self.miny))
            parent_index = n.parent_index

        return rx, ry

    def calc_position(self, index, minp):
        pos = index * self.reso + minp
        return pos

    def calc_xy_index(self, position, minp):
        return round((position - minp) / self.reso)

    def calc_index(self, node):
        return (node.y - self.miny) * self.x_width + (node.x - self.minx)

    def verify_node(self, node):
        px = self.calc_position(node.x, self.minx)
        py = self.calc_position(node.y, self.miny)

        if px < self.minx:
            return False
        if py < self.miny:
            return False
        if px >= self.maxx:
            return False
        if py >= self.maxy:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)]]

        return motion
