"""Credit to: https://github.com/ElectronicElephant/pybullet_ur5_robotiq"""

import pybullet as p
import math
import corallab_assets

from .robot_base import RobotBase


UR5_ROBOTIQ85_URDF_PATH = str(corallab_assets.get_resource_path("ur5/ur5_robotiq_85.urdf"))


class UR5Robotiq85(RobotBase):
    def __init_robot__(self, urdf_override=None):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        self.id = p.loadURDF(urdf_override or UR5_ROBOTIQ85_URDF_PATH, self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
        self.gripper_target = self.gripper_range[1]

    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id,
                                   self.id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def disable_collisions(self, objects):
        enableCollision = 0
        gripper_link_ids = [joint.id for joint in self.joints if ("pad" in joint.name or
                                                                  "finger" in joint.name or
                                                                  "knuckle" in joint.name)]
        print(objects)
        print(gripper_link_ids)

        for oid in objects:
            for l in gripper_link_ids:
                p.setCollisionFilterPair(self.id, int(oid), l, -1, enableCollision)

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=10000 or self.joints[self.mimic_parent_id].maxForce,
                                maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)
