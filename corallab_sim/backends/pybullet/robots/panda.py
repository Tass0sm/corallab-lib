"""Credit to: https://github.com/ElectronicElephant/pybullet_ur5_robotiq"""

import pybullet as p
import importlib.resources
import corallab_assets

from .robot_base import RobotBase


PANDA_URDF_PATH = str(corallab_assets.get_resource_path("panda/panda.urdf"))


class Panda(RobotBase):
    def __init_robot__(self, urdf_override=None):
        # define the robot
        # see https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py
        self.eef_id = 11
        self.arm_num_dofs = 7
        self.arm_rest_poses = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]
        self.id = p.loadURDF(urdf_override or PANDA_URDF_PATH,
                             self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.04]
        # create a constraint to keep the fingers centered
        # c = p.createConstraint(self.id,
        #                        9,
        #                        self.id,
        #                        10,
        #                        jointType=p.JOINT_GEAR,
        #                        jointAxis=[1, 0, 0],
        #                        parentFramePosition=[0, 0, 0],
        #                        childFramePosition=[0, 0, 0])
        # p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    def move_gripper(self, open_length):
        assert self.gripper_range[0] <= open_length <= self.gripper_range[1]
        for i in [9, 10]:
            p.setJointMotorControl2(self.id, i, p.POSITION_CONTROL, open_length, force=20)
