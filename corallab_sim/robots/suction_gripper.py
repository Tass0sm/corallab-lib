"""Adapted from TransporterNet code: https://arxiv.org/pdf/2010.14406.pdf"""

import numpy as np
import pybullet as p
from importlib.resources import files
from corallab_sim.utilities.bullet import load_urdf
from corallab_sim.robots.gripper import Gripper


class Suction(Gripper):
    """Simulate simple suction dynamics."""

    def __init__(self, robot, ee, obj_ids):
        """Creates suction and 'attaches' it to the robot.

        Has special cases when dealing with rigid vs deformables. For rigid,
        only need to check contact_constraint for any constraint. For soft
        bodies (i.e., cloth or bags), use cloth_threshold to check distances
        from gripper body (self.body) to any vertex in the cloth mesh. We
        need correct code logic to handle gripping potentially a rigid or a
        deformable (and similarly for releasing).

        To be clear on terminology: 'deformable' here should be interpreted
        as a PyBullet 'softBody', which includes cloths and bags. There's
        also cables, but those are formed by connecting rigid body beads, so
        they can use standard 'rigid body' grasping code.

        To get the suction gripper pose, use p.getLinkState(self.body, 0),
        and not p.getBasePositionAndOrientation(self.body) as the latter is
        about z=0.03m higher and empirically seems worse.

        Args:
          robot: int representing PyBullet ID of robot.
          ee: int representing PyBullet ID of end effector link.
          obj_ids: list of PyBullet IDs of all suctionable objects in the env.
        """
        super().__init__()

        # Load suction gripper base model (visual only).
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))

        suction_base_urdf = files("corallab_sim.robots").joinpath("assets/ur5/suction/suction-base.urdf")
        base = load_urdf(p, str(suction_base_urdf), pose[0], pose[1])
        p.createConstraint(
            parentBodyUniqueId=robot.id,
            parentLinkIndex=ee,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))

        # Load suction tip model (visual and collision) with compliance.
        # urdf = 'assets/ur5/suction/suction-head.urdf'
        pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))

        suction_head_urdf = files("corallab_sim.robots").joinpath("assets/ur5/suction/suction-head.urdf")
        self.body = load_urdf(p, str(suction_head_urdf), pose[0], pose[1])
        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot.id,
            parentLinkIndex=ee,
            childBodyUniqueId=self.body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08))
        p.changeConstraint(constraint_id, maxForce=10000)
        # p.changeConstraint(constraint_id, maxForce=50)

        # Reference to object IDs in environment for simulating suction.
        self.obj_ids = obj_ids

        # Indicates whether gripper is gripping anything (rigid or def).
        self.activated = False

        # For gripping and releasing rigid objects.
        self.contact_constraint = None

        # Defaults for deformable parameters, and can override in tasks.
        self.def_ignore = 0.035  # TODO(daniel) check if this is needed
        self.def_threshold = 0.030
        self.def_nb_anchors = 1

        # Track which deformable is being gripped (if any), and anchors.
        self.def_grip_item = None
        self.def_grip_anchors = []

        # Determines release when gripped deformable touches a rigid/def.
        # TODO(daniel) should check if the code uses this -- not sure?
        self.def_min_vetex = None
        self.def_min_distance = None

        # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
        self.init_grip_distance = None
        self.init_grip_item = None

    def activate(self):
        """Simulate suction using a rigid fixed constraint to contacted object."""
        # TODO(andyzeng): check deformables logic.
        # del def_ids

        if not self.activated:
            points = p.getContactPoints(bodyA=self.body, linkIndexA=0)
            # print(points)
            if points:

                # Handle contact between suction with a rigid object.
                for point in points:
                    obj_id, contact_link = point[2], point[4]
                # if obj_id in self.obj_ids['rigid']:
                body_pose = p.getLinkState(self.body, 0)
                obj_pose = p.getBasePositionAndOrientation(obj_id)
                world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                   world_to_body[1],
                                                   obj_pose[0], obj_pose[1])
                self.contact_constraint = p.createConstraint(
                    parentBodyUniqueId=self.body,
                    parentLinkIndex=0,
                    childBodyUniqueId=obj_id,
                    childLinkIndex=contact_link,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=obj_to_body[0],
                    parentFrameOrientation=obj_to_body[1],
                    childFramePosition=(0, 0, 0),
                    childFrameOrientation=(0, 0, 0))

                self.activated = True

    def release(self):
        """Release gripper object, only applied if gripper is 'activated'.

        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.

        To handle deformables, simply remove constraints (i.e., anchors).
        Also reset any relevant variables, e.g., if releasing a rigid, we
        should reset init_grip values back to None, which will be re-assigned
        in any subsequent grasps.
        """
        if self.activated:
            self.activated = False

            # Release gripped rigid object (if any).
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:  # pylint: disable=bare-except
                    pass
                self.init_grip_distance = None
                self.init_grip_item = None

            # Release gripped deformable object (if any).
            if self.def_grip_anchors:
                for anchor_id in self.def_grip_anchors:
                    p.removeConstraint(anchor_id)
                self.def_grip_anchors = []
                self.def_grip_item = None
                self.def_min_vetex = None
                self.def_min_distance = None

    def detect_contact(self):
        """Detects a contact with a rigid object."""
        body, link = self.body, 0
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:  # pylint: disable=bare-except
                self.contact_constraint = None
                pass

        # Get all contact points between the suction and a rigid body.
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        # print(points)
        # exit()
        if self.activated:
            points = [point for point in points if point[2] != self.body]

        # # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True

        return False

    def check_grasp(self):
        """Check a grasp (object in contact?) for picking success."""

        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
        return suctioned_object

    def grasp(self, on, colmask):
        # collision masks for simplifying testing
        obj = self.check_grasp()

        if on:
            self.activate()
            if colmask and obj is not None:
                # set objs base link to not collide with anything?
                p.setCollisionFilterGroupMask(obj, -1, 0, 0)
        else:
            if colmask and obj is not None:
                # set objs base link to collide with everything?
                p.setCollisionFilterGroupMask(obj, -1, 1, 1)
            self.release()
