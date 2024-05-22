"""Credit to: https://github.com/lyfkyle/pybullet_ompl/"""

import os.path as osp
import pybullet as p
import math
import sys

import corallab_lib
from corallab_lib import Robot, Env, Task
from corallab_lib.backends.pybullet import draw_frame
from corallab_lib.backends.pybullet.ompl import pb_ompl

corallab_lib.backend_manager.set_backend("pybullet")


class BoxDemo():
    def __init__(self):
        robot = Robot("UR5")
        env = Env(add_plane=True)
        self.task = Task(robot=robot, env=env)

        self.pc = env.env_impl.client
        self.robot = robot.robot_impl.robot_impl
        self.obstacles = []

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("BITstar")

        # add obstacles
        self.add_obstacles()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            self.pc.removeBody(obstacle)

    def add_obstacles(self):
        # add box
        self.add_box([1, 0, 0.7], [0.5, 0.5, 0.05])

        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        colBoxId = self.pc.createCollisionShape(self.pc.GEOM_BOX, halfExtents=half_box_size)
        box_id = self.pc.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)

        self.obstacles.append(box_id)
        return box_id

    def demo(self):
        # start = [0, 0, 0, -1, 0, 1.5, 0]
        # goal = [0, 1.5, 0, -0.1, 0, 0.2, 0]
        start = [0, 0, 0, -1, 0, 1.5]
        goal = [0, 1.5, 0, -0.1, 0, 0.2]

        self.robot.set_state(start)
        res, path = self.pb_ompl_interface.plan(goal)
        if res:
            self.pb_ompl_interface.execute(path)
        return res, path

if __name__ == '__main__':
    env = BoxDemo()
    env.demo()
