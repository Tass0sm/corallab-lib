"""Credit to: https://github.com/ElectronicElephant/pybullet_ur5_robotiq"""

import pybullet as pb
import math
import numpy as np
import corallab_assets

from corallab_lib import Robot
from .robot_base import RobotBase
from .triple_ur5e import TripleUR5e

TRIPLE_UR5_ASSET_PATH = str(corallab_assets.get_resource_path("triple_ur5"))
TRIPLE_UR5_URDF_PATH = str(corallab_assets.get_resource_path("triple_ur5/triple_ur5e_with_grippers.urdf"))


class TripleUR5eWithGrippers(TripleUR5e):
    def __init_robot__(self, p=pb, urdf_override=None):
        self.eef_id = 6
        self.arm_num_dofs = 18
        self.arm_rest_poses = [
            0.0000, -2.2000, 1.9000, -1.3830, -1.5700,  0.0000,
            0.0000, -2.2000, 1.9000, -1.3830, -1.5700,  0.0000,
            0.0000, -2.2000, 1.9000, -1.3830, -1.5700, 0.0000
        ]

        self._p = p
        self._p.setAdditionalSearchPath(TRIPLE_UR5_ASSET_PATH)
        self.id = self._p.loadURDF(urdf_override or TRIPLE_UR5_URDF_PATH, self.base_pos, self.base_ori,
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
        mimic_parent_name_0 = 'finger_joint_0'
        mimic_children_names_0 = {'right_outer_knuckle_joint_0': 1,
                                  'left_inner_knuckle_joint_0': 1,
                                  'right_inner_knuckle_joint_0': 1,
                                  'left_inner_finger_joint_0': -1,
                                  'right_inner_finger_joint_0': -1}
        self.mimic_joint0_id = self.__setup_mimic_joints__(mimic_parent_name_0, mimic_children_names_0)

        mimic_parent_name_1 = 'finger_joint_1'
        mimic_children_names_1 = {'right_outer_knuckle_joint_1': 1,
                                  'left_inner_knuckle_joint_1': 1,
                                  'right_inner_knuckle_joint_1': 1,
                                  'left_inner_finger_joint_1': -1,
                                  'right_inner_finger_joint_1': -1}
        self.mimic_joint1_id = self.__setup_mimic_joints__(mimic_parent_name_1, mimic_children_names_1)

        mimic_parent_name_2 = 'finger_joint_2'
        mimic_children_names_2 = {'right_outer_knuckle_joint_2': 1,
                                  'left_inner_knuckle_joint_2': 1,
                                  'right_inner_knuckle_joint_2': 1,
                                  'left_inner_finger_joint_2': -1,
                                  'right_inner_finger_joint_2': -1}
        self.mimic_joint2_id = self.__setup_mimic_joints__(mimic_parent_name_2, mimic_children_names_2)

        self.ee_link_0_id = [joint.id for joint in self.joints if joint.name == "flange-ee_link_0"][0]
        self.ee_link_1_id = [joint.id for joint in self.joints if joint.name == "flange-ee_link_1"][0]
        self.ee_link_2_id = [joint.id for joint in self.joints if joint.name == "flange-ee_link_2"][0]

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

        self._p.setJointMotorControl2(self.id, self.mimic_joint2_id, pb.POSITION_CONTROL, targetPosition=open_angle,
                                      force=10000 or self.joints[self.mimic_joint2_id].maxForce,
                                      maxVelocity=self.joints[self.mimic_joint2_id].maxVelocity)

    def fake_open_grippers(self, ids):
        self.gripper_target = self.gripper_range[0] + 0.05
        self.move_gripper(self.gripper_target)
        self.fake_open_gripper0(ids[0])
        self.fake_open_gripper1(ids[1])
        self.fake_open_gripper2(ids[2])

    def fake_open_gripper0(self, id):
        self._p.removeConstraint(self.gripper0_constraint_id)

    def fake_open_gripper1(self, id):
        self._p.removeConstraint(self.gripper1_constraint_id)

    def fake_open_gripper2(self, id):
        self._p.removeConstraint(self.gripper2_constraint_id)

    def fake_close_grippers(self, ids):
        self.gripper_target = self.gripper_range[0] + 0.025
        self.move_gripper(self.gripper_target)
        self.fake_close_gripper0(ids[0])
        self.fake_close_gripper1(ids[1])
        self.fake_close_gripper2(ids[2])

    def fake_close_gripper0(self, id):
        self.gripper0_constraint_id = self.fake_close_gripper(self.ee_link_0_id, id)

    def fake_close_gripper1(self, id):
        self.gripper1_constraint_id = self.fake_close_gripper(self.ee_link_1_id, id)

    def fake_close_gripper2(self, id):
        self.gripper2_constraint_id = self.fake_close_gripper(self.ee_link_2_id, id)

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
        base_link_1_idx = 38

        base_pos_0 = self._p.getLinkState(self.id, base_link_0_idx)[0]
        base_pos_1 = self._p.getLinkState(self.id, base_link_1_idx)[0]
        base_pos_2 = self._p.getLinkState(self.id, base_link_2_idx)[0]

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

        UR5_2 = Robot(
            "UR5",
            pos=base_pos_2,
            backend="pybullet"
        )
        UR5_2.set_id("UR5_2")

        return [UR5_0, UR5_1, UR5_2]