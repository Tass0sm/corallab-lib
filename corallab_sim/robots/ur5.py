"""Robot class to streamline controlling the simulated UR5 robot"""

import os
from typing import Iterable
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
import time

# import rtde.rtde as rtde
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from .robotiq_gripper import RobotiqGripper

from corallab_sim.robots.suction_gripper import Suction
from corallab_sim.utilities.spatial import get_transform, get_rotation, transform_point, invert_transform
from abc import ABC, abstractmethod
from importlib.resources import files


def log_info(gripper):
    print(f"Pos: {str(gripper.get_current_position()): >3}  "
          f"Open: {gripper.is_open(): <2}  "
          f"Closed: {gripper.is_closed(): <2}  ")


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


class RealUR5(UR5):
    def __init__(self, ip_address, base_pos=(0, 0, 0), base_orn=(0, 0, 0, 1), gripper=None, move_timestep=0):
        """
        base_pos - position of robot base in world frame
        base_orn - orientation of robot base in world frame
        """
        self.con = RTDEControlInterface(ip_address)
        self.rec = RTDEReceiveInterface(ip_address)
        self.con.setTcp([0, 0, 0.145, 0, 0, 0])

        self.gripper = gripper
        self.gripper.connect(ip_address, 63352)
        self.gripper.activate()

        self.base_pos = base_pos
        self.base_orn = base_orn
        self.base_transform = get_transform(rotq=base_orn, pos=base_pos)
        self.move_timestep = move_timestep

    def test(self):
        actual_q = self.rec.getActualQ();

        position1 = [-0.343, -0.435, 0.50, -0.001, 3.12, 0.04];
        position2 = [-0.243, -0.335, 0.20, -0.001, 3.12, 0.04];
        
        self.con.moveL(position1);
        self.con.moveL(position2);
        self.con.stopL(10.0, False);

        if self.gripper is not None:
            print("Testing gripper...")
            self.gripper.move_and_wait_for_pos(255, 255, 255)
            log_info(self.gripper)
            self.gripper.move_and_wait_for_pos(0, 255, 255)
            log_info(self.gripper)

    def go_home(self):
        pass

    def enter_teach_mode(self):
        self.con.teachMode()

    def exit_teach_mode(self):
        self.con.endTeachMode()

    def convert_pose_to_base_frame(self, pos, rot):
        """convert position (meters given in the world frame) and rot (a scipy
        spatial rotation) to a pose (x, y, z, rx, ry, rz) in the robot's base
        frame

        T_w * p = T_b * p'
        T_b^(-1) * T_w * p = p'
        T_b^(-1) * I * p = p'
        T_b^(-1) * p = p'

        """

        T_b_inv = invert_transform(self.base_transform)
        R_b_inv = get_rotation(t_matrix=T_b_inv)

        # change basis of pos
        pos_B = transform_point(T_b_inv, pos)

        # change basis of orn
        new_rot = rot * R_b_inv
        rotvec = new_rot.as_rotvec(degrees=False)

        # combine
        pose = np.hstack([pos, rotvec])
        return pose

    def get_joint_positions(self):
        return self.rec.getActualQ()

    def ee_pose(self):
        fk = self.con.getForwardKinematics(self.rec.getActualQ(), self.con.getTCPOffset())

        pos = fk[0:3]
        rot = R.from_rotvec(fk[3:6], degrees=False)
        q = rot.as_quat()

        return pos, q

    def move_ee(self, pos, orn):
        """Move end effector to position POS (meters given in the world frame)
        with orientation ORN (a quaternion).

        In pybullet the orientations of the end effector are left-handed and
        defined with Z pointing into the robot. That way, you can give an
        object's orientation and the robot will assume an orientation that can
        interface that object from the top. On the robot, this frame is
        apparently flipped upside down. So we flip the input orientation by 180
        degrees about its x-axis to get something that matches this convention.
        """

        rot = R.from_quat(orn)
        rot_x_180 = R.from_euler("xyz", [180, 0, 0], degrees=True)
        new_rot = rot * rot_x_180

        pose_base = self.convert_pose_to_base_frame(pos, new_rot)
        self.con.moveL(pose_base)

    def move_ee_down(self, pos, orn=(1, 0, 0, 0), **kwargs):
        """
        moves down from `pos` to z=0 until it detects object
        returns: pose=(pos, orn) at which it detected contact"""
        speed = 0.1
        assert speed > 0
        # self.con.moveUntilContact([0, 0, -speed, 0, 0, 0])
        return self.ee_pose()

    def move_ee_above(self, pos, orn=(1, 0, 0, 0), above_offt=(0, 0, 0.2)):
        a_pos = np.add(pos, above_offt)
        self.move_ee(a_pos, orn)

    def get_gripper_state(self):
        if self.gripper is not None:
            return self.gripper.get_state()

    def toggle_gripper(self):
        if self.gripper is not None:
            return self.gripper.toggle()

    def __del__(self):
        self.con.stopScript()
        print("Stopping robot control script")


class SimulatedUR5(UR5):
    def __init__(self, base_pos, orn=(0, 0, 0), move_timestep=0):
        """
        base_pos - position of robot base in world frame
        """

        ur5_urdf_path = files("corallab_sim.robots").joinpath("assets/ur5/ur5.urdf")
        self.id = p.loadURDF(str(ur5_urdf_path), base_pos, p.getQuaternionFromEuler(orn))

        ddict = {'fixed': [], 'rigid': [], 'deformable': []}
        self.ee_id = 10
        self.ee = Suction(self.id, self.ee_id-1, ddict)

        self.n_joints = p.getNumJoints(self.id)
        joints = [p.getJointInfo(self.id, i) for i in range(self.n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        self.home_q = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.set_q(self.home_q)

        self.ee.release()
        self.move_timestep = move_timestep

    def set_q(self, q):
        for ji, qi in zip(self.joints, q):
            p.resetJointState(self.id, ji, qi)

    def ik(self, pos, orn):
        """ Written with help of TransporterNet code: https://arxiv.org/pdf/2010.14406.pdf"""

        # flip this orientation to match the conventions of the real robot
        rot = R.from_quat(orn)
        rot_x_180 = R.from_euler("xyz", [180, 0, 0], degrees=True)
        new_rot = rot * rot_x_180
        new_orn = new_rot.as_quat()


        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=self.ee_id,
            targetPosition=pos,
            targetOrientation=new_orn,
            lowerLimits=[-3*np.pi/2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi/2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.home_q).tolist(),
            maxNumIterations=200,
            residualThreshold=1e-5)
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

    def move_q(self, tar_q, error_thresh=1e-2, speed=0.01, break_cond=lambda: False, max_iter=300, **kwargs):
        """ Written with help of TransporterNet code: https://arxiv.org/pdf/2010.14406.pdf"""
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
                positionGains=np.ones(len(self.joints)))
            p.stepSimulation()
            i += 1
            time.sleep(self.move_timestep)

        # p.removeBody(marker)
        return False, tar_q, cur_q

    def move_ee(self, pos, orn=None, error_thresh=1e-2, speed=0.01, break_cond=lambda: False, max_iter=300, **kwargs):
        tar_q = self.ik(pos, orn)
        # marker = draw_sphere_marker(pos)
        self.move_q(tar_q, error_thresh=error_thresh, speed=speed, break_cond=break_cond, max_iter=max_iter, **kwargs)

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

    def suction(self, on, colmask=True):
        # collision masks for simplifying testing
        obj = self.ee.check_grasp()

        if on:
            self.ee.activate()
            if colmask and obj is not None:
                p.setCollisionFilterGroupMask(obj, -1, 0, 0)
        else:
            if colmask and obj is not None:
                p.setCollisionFilterGroupMask(obj, -1, 1, 1)
            self.ee.release()

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


def teach_and_record_trajectory(robot):
    """Set robot to freedrive mode, and repeatedly prompt user to set positions
    for a trajectory. Return the recorded list of poses."""

    poses = []
    configurations = []
    robot.enter_teach_mode()

    print("- Robot is now in free-drive mode.")
    print("- Move the robot to the desired positions.")
    print("- Record a pose with 'r' + RET")
    print("- Toggle the gripper with 't' + RET")
    print("- Quit the loop with 'q' + RET")

    i = 0
    while True:
        command = input("> ")

        if command == "r":
            pose = robot.ee_pose()
            gripper_state = robot.get_gripper_state()
            poses.append((*pose, gripper_state))
            print(f"Recorded position {i}")

            config = robot.get_joint_positions()
            configurations.append(config)
            print(f"Recorded configuration {i}")

            i += 1
        elif command == "t":
            new_state = robot.toggle_gripper()
            print(f"Gripper is now in state {new_state}")
        elif command == "q":
            break
        else:
            print("Unknown command")

    print(f"Recorded {i} poses")

    robot.exit_teach_mode()

    return poses, configurations
