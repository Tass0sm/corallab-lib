"""Credit to: https://github.com/ElectronicElephant/pybullet_ur5_robotiq"""

import copy
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import namedtuple


def unit_vec(lst: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(lst)
    return (lst / mag) if mag > 0 else 0


class RobotBase(object):
    """
    The base class for robots
    """

    def __init__(self, pos, ori, target_trans=[0, 0, 0], target_rot=[0, 0, 0]):
        """
        Arguments:
            pos: [x y z]
            ori: [r p y]
            target_rot: [rx, ry, rz]

        Attributes:
            id: Int, the ID of the robot
            eef_id: Int, the ID of the End-Effector
            arm_num_dofs: Int, the number of DoFs of the arm
                i.e., the IK for the EE will consider the first `arm_num_dofs` controllable (non-Fixed) joints
            joints: List, a list of joint info
            controllable_joints: List of Ints, IDs for all controllable joints
            arm_controllable_joints: List of Ints, IDs for all controllable joints on the arm (that is, the first `arm_num_dofs` of controllable joints)

            ---
            For null-space IK
            ---
            arm_lower_limits: List, the lower limits for all controllable joints on the arm
            arm_upper_limits: List
            arm_joint_ranges: List
            arm_rest_poses: List, the rest position for all controllable joints on the arm

            gripper_range: List[Min, Max]
        """
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.target_rot = target_rot
        self.target_trans = target_trans

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        print(self.joints)

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo',
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]

    def __init_robot__(self):
        raise NotImplementedError

    def __post_load__(self):
        pass

    def reset(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        reset to rest poses
        """
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.gripper_target = self.gripper_range[1]
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.gripper_target = self.gripper_range[0]
        self.move_gripper(self.gripper_range[0])

    @property
    def ee_pose(self):
        return p.getLinkState(self.id, self.eef_id)[:2]

    def move_ee(self, action, control_method):
        assert control_method in ('joint', 'end')
        if control_method == 'end':
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses,
                                                       maxNumIterations=20)
        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            joint_poses = action
        # arm
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)

    def move_gripper(self, open_length):
        raise NotImplementedError

    def get_joint_obs(self):
        positions = []
        velocities = []
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)

    ########################
    #     GET / SET q      #
    ########################

    def get_q(self):
        cur_q = np.array([p.getJointState(self.id, i)[0] for i in self.arm_controllable_joints])
        return cur_q

    def set_q(self, q):
        for ji, qi in zip(self.arm_controllable_joints, q):
            p.resetJointState(self.id, ji, qi, targetVelocity=0)

    def go_home(self):
        self.set_q(self.arm_rest_poses)

    ############################
    # SYNCHRONOUS ARM MOVEMENT #
    ############################

    def ik(self, pos, orn, max_niter=200):
        """Written with help of TransporterNet code: https://arxiv.org/pdf/2010.14406.pdf"""

        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=self.eef_id,
            targetPosition=pos,
            targetOrientation=orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
            maxNumIterations=max_niter,
            residualThreshold=1e-5,
        )
        joints = np.float64(joints)
        return joints

    def move_q_synchronous(
            self,
            tar_q,
            error_thresh=1e-2,
            speed=0.01,
            break_cond=lambda: False,
            max_iter=5000,
            **kwargs,
    ):
        """Written with help of TransporterNet code: https://arxiv.org/pdf/2010.14406.pdf"""
        i = 0
        assert i < max_iter
        while i < max_iter:
            cur_q = self.get_q()
            err_q = tar_q - cur_q
            if break_cond() or (np.abs(err_q) < error_thresh).all():
                return True, tar_q, cur_q

            u = unit_vec(err_q)
            step_q = cur_q + u * speed

            p.setJointMotorControlArray(
                bodyIndex=self.id,
                jointIndices=self.arm_controllable_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_q,
                positionGains=np.ones(len(self.arm_controllable_joints)),
            )

            self.move_gripper(self.gripper_target)

            self.step_simulation()
            i += 1

        return False, tar_q, cur_q

    def move_ee_synchronous(self, pos, orn=None, **kwargs):
        tar_q = self.ik(pos, orn)
        tar_q = tar_q[:self.arm_num_dofs]
        self.move_q_synchronous(tar_q, **kwargs)

    def move_ee_above_synchronous(self, pos, orn=(1, 0, 0, 0), above_offt=(0, 0, 0.2)):
        a_pos = np.add(pos, above_offt)
        self.move_ee_synchronous(a_pos, orn)

    def dont_move(self, niters=200):
        cur_q = self.get_q()
        for i in range(niters):
            # print(i)
            # self.move_ee(cur_q, "joint")
            self.set_q(cur_q)
            self.move_gripper(self.gripper_target)
            self.step_simulation()

    ####
    # Util
    ####

    def convert_target_pose(self, pos, orn):
        pos = np.array(pos) + self.target_trans

        rot = R.from_quat(orn)
        rot_x_180 = R.from_euler("xyz", self.target_rot, degrees=True)
        rot = rot * rot_x_180
        orn = rot.as_quat()

        return pos, orn


    ####
    # For PB_OMPL
    ####

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self.set_q(state)
        self.state = state

    def get_cur_state(self):
        return copy.deepcopy(self.state)
