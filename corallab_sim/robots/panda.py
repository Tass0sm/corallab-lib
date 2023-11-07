import time
import numpy as np
import pybullet as p
from typing import Iterable
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R


class Panda(ABC):
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


class SimulatedPanda(Panda):
    def __init__(self, base_pos, orn=(0, 0, 0), move_timestep=0, flipping=True):
        """
        base_pos - position of robot base in world frame
        """

        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.id = p.loadURDF("franka_panda/panda.urdf", base_pos, p.getQuaternionFromEuler(orn), useFixedBase=True, flags=flags)

        self.n_joints = p.getNumJoints(self.id)
        joints = [p.getJointInfo(self.id, i) for i in range(self.n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        self.pandaEndEffectorIndex = 11

        pandaNumDofs = 7
        self.pandaNumDofs = pandaNumDofs
        self.lower_limits = [-7] * pandaNumDofs
        self.upper_limits = [7] * pandaNumDofs
        self.joint_ranges = [7] * pandaNumDofs
        self.home_q = np.array([0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02])
        self.set_q(self.home_q)

        self.move_timestep = move_timestep
        self.flipping = flipping

        # create a constraint to keep the fingers centered
        c = p.createConstraint(self.id,
                               9,                        # finger
                               self.id,
                               10,                       # finger
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        index = 0
        for j in range(p.getNumJoints(self.id)):
            p.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.id, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC):
                p.resetJointState(self.id, j, self.home_q[index])
                index += 1

            if (jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self.id, j, self.home_q[index])
                index += 1

    def set_q(self, q):
        for ji, qi in zip(self.joints, q):
            p.resetJointState(self.id, ji, qi)

    def ik(self, pos, orn, max_iters):
        """ Written with help of TransporterNet code: https://arxiv.org/pdf/2010.14406.pdf"""

        if self.flipping:
            # flip this orientation to match the conventions of the real robot
            rot = R.from_quat(orn)
            rot_x_180 = R.from_euler("xyz", [180, 0, 0], degrees=True)
            rot = rot * rot_x_180
            orn = rot.as_quat()

        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=self.pandaEndEffectorIndex,
            targetPosition=pos,
            targetOrientation=orn,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=np.float32(self.home_q).tolist(),
            maxNumIterations=max_iters,
            residualThreshold=1e-5)

        # joints = np.float32(joints)
        # joints[2:] = (joints[2:]+np.pi) % (2*np.pi) - np.pi
        return joints[:self.pandaNumDofs]

    @classmethod
    def _norm(cls, it: Iterable) -> float:
        return np.linalg.norm(it)

    @classmethod
    def _unit_vec(cls, lst: np.ndarray) -> np.ndarray:
        mag = cls._norm(lst)
        return (lst / mag) if mag > 0 else 0

    def move_q(self, tar_q, error_thresh=1e-2, speed=0.01, break_cond=lambda: False, max_iter=10000, **kwargs):
        """Written with help of TransporterNet code:
        https://arxiv.org/pdf/2010.14406.pdf"""
        i = 0
        assert i < max_iter
        while i < max_iter:
            cur_q = np.array([p.getJointState(self.id, i)[0] for i in self.joints])
            err_q = tar_q - cur_q

            if break_cond() or (np.abs(err_q) < error_thresh).all():
                print(err_q)
                return True, tar_q, cur_q

            u = self._unit_vec(err_q)
            step_q = cur_q + u * speed
            p.setJointMotorControlArray(
                bodyIndex=self.id,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_q,
                positionGains=np.ones(len(self.joints)))
            p.stepSimulation()
            i += 1
            time.sleep(self.move_timestep)

        return False, tar_q, cur_q

    def move_q2(self, pos, orn, error_thresh=1e-2, speed=0.01, break_cond=lambda: False, max_iter=10000, **kwargs):
        """Written with help of TransporterNet code:
        https://arxiv.org/pdf/2010.14406.pdf"""
        i = 0
        assert i < max_iter
        while i < max_iter:
            cur_q = np.array([p.getJointState(self.id, i)[0] for i in self.joints])
            tar_q = self.ik(pos, orn, 10)
            err_q = tar_q - cur_q

            if break_cond() or (np.abs(err_q) < error_thresh).all():
                print(err_q)
                return True, tar_q, cur_q

            u = self._unit_vec(err_q)
            step_q = cur_q + u * speed
            p.setJointMotorControlArray(
                bodyIndex=self.id,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_q,
                positionGains=np.ones(len(self.joints)))
            p.stepSimulation()
            i += 1
            time.sleep(self.move_timestep)

        return False, tar_q, cur_q

    def move_ee_old(self, pos, orn=None, **kwargs):
        tar_q = self.ik(pos, orn, 200)
        self.move_q(tar_q, **kwargs)

    def move_ee(self, pos, orn=None, **kwargs):
        self.move_q2(pos, orn, **kwargs)

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
        if on:
            finger_target = 0.01
        else:
            finger_target = 0.04

        for iteration in range(100):
            for i in [9, 10]:
                p.setJointMotorControl2(self.id,
                                        i,
                                        p.POSITION_CONTROL,
                                        finger_target,
                                        force=20)
            p.stepSimulation()
            time.sleep(self.move_timestep)

    def set_joints(self, q):
        for ji, qi in zip(self.joints, q):
            p.resetJointState(self.id, ji, qi)

    @property
    def ee_pose(self):
        return p.getLinkState(self.id, self.pandaEndEffectorIndex)[:2]

    @property
    def ee_offset(self):
        return p.getLinkState(self.id, self.pandaEndEffectorIndex)[2:4]

    @property
    def ee_frame(self):
        return p.getLinkState(self.id, self.pandaEndEffectorIndex)[4:6]

    def go_home(self):
        self.set_q(self.home_q)
