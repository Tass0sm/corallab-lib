"""Robot class to streamline controlling the simulated UR5 robot"""

import os
import sys
from typing import Iterable
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
import time

# import rtde.rtde as rtde
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# from corallab_sim.robots.robotiq_gripper import RobotiqGripper

from corallab_sim.using_pybullet.robots.suction_gripper import Suction
from corallab_sim.utilities.spatial import (
    get_transform,
    get_rotation,
    transform_point,
    invert_transform,
)
from abc import ABC, abstractmethod
from importlib.resources import files


def log_info(gripper):
    print(
        f"Pos: {str(gripper.get_current_position()): >3}  "
        f"Open: {gripper.is_open(): <2}  "
        f"Closed: {gripper.is_closed(): <2}  "
    )


class UR5(ABC):
    """Abstract class defining API for controlling a robot"""

    # @abstractmethod
    # def move_ee(self, pos, orn=None, error_thresh=1e-2, speed=0.01, break_cond=lambda: False, max_iter=300, **kwargs):
    #     """Move end effector to position POS with orientation ORN."""
    #     pass

    @abstractmethod
    def move_ee_down(self, pos, orn=(0, 0, 0, 1)):
        pass

    @abstractmethod
    def move_ee_above(self, pos, orn=(0, 0, 0, 1), above_offt=(0, 0, 0.2)):
        pass


class SimulatedUR5(UR5):
    def __init__(
        self,
        base_pos,
        orn=(0, 0, 0),
        move_timestep=0,
        flipping=True,
        GripperClass=Suction,
    ):
        """
        base_pos - position of robot base in world frame
        """

        ur5_urdf_path = files("corallab_sim.robots").joinpath("assets/ur5/ur5.urdf")
        self.id = p.loadURDF(
            str(ur5_urdf_path), base_pos, p.getQuaternionFromEuler(orn)
        )

        self.n_joints = p.getNumJoints(self.id)
        joints = [p.getJointInfo(self.id, i) for i in range(self.n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        self.home_q = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.set_q(self.home_q)

        self.move_timestep = move_timestep
        self.flipping = flipping

        ddict = {"fixed": [], "rigid": [], "deformable": []}
        self.ee_id = 10
        self.ee = GripperClass(self, self.ee_id - 1, ddict)
        self.ee.release()

    def set_q(self, q):
        for ji, qi in zip(self.joints, q):
            p.resetJointState(self.id, ji, qi)

    def ik(self, pos, orn):
        """Written with help of TransporterNet code: https://arxiv.org/pdf/2010.14406.pdf"""

        if self.flipping:
            # flip this orientation to match the conventions of the real robot
            rot = R.from_quat(orn)
            rot_x_180 = R.from_euler("xyz", [180, 0, 0], degrees=True)
            rot = rot * rot_x_180
            orn = rot.as_quat()

        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=self.ee_id,
            targetPosition=pos,
            targetOrientation=orn,
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.home_q).tolist(),
            maxNumIterations=200,
            residualThreshold=1e-5,
        )
        joints = np.float32(joints)
        # joints[2:] = (joints[2:]+np.pi) % (2*np.pi) - np.pi
        return joints

    @classmethod
    def _norm(cls, it: Iterable) -> float:
        return np.linalg.norm(it)

    @classmethod
    def _unit_vec(cls, lst: np.ndarray) -> np.ndarray:
        mag = cls._norm(lst)
        return (lst / mag) if mag > 0 else 0

    def move_q(
        self,
        tar_q,
        error_thresh=1e-2,
        speed=0.01,
        break_cond=lambda: False,
        max_iter=10000,
        **kwargs,
    ):
        """Written with help of TransporterNet code: https://arxiv.org/pdf/2010.14406.pdf"""
        i = 0
        assert i < max_iter
        while i < max_iter:
            cur_q = np.array([p.getJointState(self.id, i)[0] for i in self.joints])
            err_q = tar_q - cur_q
            if break_cond() or (np.abs(err_q) < error_thresh).all():
                # p.removeBody(marker)
                return True, tar_q, cur_q

            u = self._unit_vec(err_q)
            step_q = cur_q + u * speed
            p.setJointMotorControlArray(
                bodyIndex=self.id,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_q,
                positionGains=np.ones(len(self.joints)),
            )
            p.stepSimulation()
            i += 1
            time.sleep(self.move_timestep)

        # p.removeBody(marker)
        return False, tar_q, cur_q

    def move_ee(self, pos, orn=None, **kwargs):
        tar_q = self.ik(pos, orn)
        # marker = draw_sphere_marker(pos)
        self.move_q(tar_q, **kwargs)

    def move_ee_down(self, pos, orn=(0, 0, 0, 1), **kwargs):
        """
        moves down from `pos` to z=0 until it detects object
        returns: pose=(pos, orn) at which it detected contact"""
        pos = [*pos[:2], 0]
        self.move_ee(pos, orn=orn, break_cond=self.ee.detect_contact, **kwargs)
        return self.ee_pose

    def move_ee_above(self, pos, orn=(0, 0, 0, 1), above_offt=(0, 0, 0.2), **kwargs):
        a_pos = np.add(pos, above_offt)
        self.move_ee(a_pos, orn=orn, **kwargs)

    def move_ee_away(self, offt):
        ee_pos, ee_orn = self.ee_pose
        target_pos = np.add(ee_pos, offt)
        self.move_ee(target_pos, ee_orn)

    def grasp(self, on, colmask=True):
        self.ee.grasp(on, colmask)

    def set_joints(self, q):
        for ji, qi in zip(self.joints, q):
            p.resetJointState(self.id, ji, qi)

    @property
    def ee_pose(self):
        return p.getLinkState(self.id, self.ee_id)[:2]

    @property
    def ee_offset(self):
        return p.getLinkState(self.id, self.ee_id)[2:4]

    @property
    def ee_frame(self):
        return p.getLinkState(self.id, self.ee_id)[4:6]

    def go_home(self):
        self.set_q(self.home_q)
