"""Credit to: https://github.com/ElectronicElephant/pybullet_ur5_robotiq"""

import pybullet as pb
import math
import numpy as np
import corallab_assets

from corallab_lib import Robot
from .robot_base import RobotBase
from .dual_ur5 import DualUR5

DUAL_UR5_ASSET_PATH = str(corallab_assets.get_resource_path("dual_ur5"))
DUAL_UR5_URDF_PATH = str(corallab_assets.get_resource_path("dual_ur5/dual_ur5_with_grippers.urdf"))


class DualUR5WithGrippers(DualUR5):
    def __init_robot__(self, p=pb, urdf_override=None):
        self.eef_id = 6
        self.arm_num_dofs = 12
        self.arm_rest_poses = [
            -1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699, -1.5707970583733368, 0.0009377758247187636,
            -1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699, -1.5707970583733368, 0.0009377758247187636
        ]

        self._p = p
        self._p.setAdditionalSearchPath(DUAL_UR5_ASSET_PATH)
        self.id = self._p.loadURDF(urdf_override or DUAL_UR5_URDF_PATH, self.base_pos, self.base_ori,
                                   useFixedBase=True, flags=pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
        self.gripper_target = self.gripper_range[1]


    def controllable_joint_mask(self, joint_name, info):
        # Mask finger and knuckle joints. Consider them uncontrollable.
        if ("finger" in joint_name or
            "knuckle" in joint_name):
            return True
        else:
            return False

    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint_0'
        mimic_children_names = {'right_outer_knuckle_joint_0': 1,
                                'left_inner_knuckle_joint_0': 1,
                                'right_inner_knuckle_joint_0': 1,
                                'left_inner_finger_joint_0': -1,
                                'right_inner_finger_joint_0': -1}
        self.mimic_joint0_id = self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

        mimic_parent_name = 'finger_joint_1'
        mimic_children_names = {'right_outer_knuckle_joint_1': 1,
                                'left_inner_knuckle_joint_1': 1,
                                'right_inner_knuckle_joint_1': 1,
                                'left_inner_finger_joint_1': -1,
                                'right_inner_finger_joint_1': -1}
        self.mimic_joint1_id = self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

        self.ee_link_0_id = [joint.id for joint in self.joints if joint.name == "ee_fixed_joint_0"][0]
        self.ee_link_1_id = [joint.id for joint in self.joints if joint.name == "ee_fixed_joint_1"][0]

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in mimic_child_multiplier.items():
            c = self._p.createConstraint(self.id, mimic_parent_id,
                                         self.id, joint_id,
                                         jointType=pb.JOINT_GEAR,
                                         jointAxis=[0, 1, 0],
                                         parentFramePosition=[0, 0, 0],
                                         childFramePosition=[0, 0, 0])
            self._p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

        return mimic_parent_id

    def move_gripper(self, open_length):
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        self._p.setJointMotorControl2(self.id, self.mimic_joint0_id, pb.POSITION_CONTROL, targetPosition=open_angle,
                                      force=10000 or self.joints[self.mimic_joint0_id].maxForce,
                                      maxVelocity=self.joints[self.mimic_joint0_id].maxVelocity)

        self._p.setJointMotorControl2(self.id, self.mimic_joint1_id, pb.POSITION_CONTROL, targetPosition=open_angle,
                                      force=10000 or self.joints[self.mimic_joint1_id].maxForce,
                                      maxVelocity=self.joints[self.mimic_joint1_id].maxVelocity)

    def fake_open_grippers(self, ids):
        self.fake_open_gripper0(ids[0])
        self.fake_open_gripper1(ids[1])

    def fake_open_gripper0(self, id):
        self._p.removeConstraint(self.gripper0_constraint_id)

    def fake_open_gripper1(self, id):
        self._p.removeConstraint(self.gripper1_constraint_id)

    def fake_close_grippers(self, ids):
        self.fake_close_gripper0(ids[0])
        self.fake_close_gripper1(ids[1])

    def fake_close_gripper0(self, id):
        self.gripper0_constraint_id = self.fake_close_gripper(self.ee_link_0_id, id)

    def fake_close_gripper1(self, id):
        self.gripper1_constraint_id = self.fake_close_gripper(self.ee_link_1_id, id)

    def fake_close_gripper(self, link_id, object_id):
        link_pose = self._p.getLinkState(self.id, link_id)
        object_pos, object_orn = self._p.getBasePositionAndOrientation(object_id)
        world_to_body = self._p.invertTransform(link_pose[0], link_pose[1])
        obj_to_body = self._p.multiplyTransforms(world_to_body[0],
                                                 world_to_body[1],
                                                 object_pos, object_orn)

        cid = self._p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=link_id,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=obj_to_body[0],
            parentFrameOrientation=obj_to_body[1],
            childFramePosition=(0, 0, 0),
            childFrameOrientation=(0, 0, 0)
        )

        return cid
    
    # Multi-Agent API
    def get_subrobots(self):
        base_link_0_idx = 0
        base_link_1_idx = 19

        base_pos_0 = self._p.getLinkState(self.id, base_link_0_idx)[0]
        base_pos_1 = self._p.getLinkState(self.id, base_link_1_idx)[0]

        UR5_0 = Robot(
            "UR5",
            pos=base_pos_0,
            backend="pybullet"
        )
        UR5_0.set_id("UR5_0")

        UR5_1 = Robot(
            "UR5",
            pos=base_pos_1,
            backend="pybullet"
        )
        UR5_1.set_id("UR5_1")

        return [UR5_0, UR5_1]
