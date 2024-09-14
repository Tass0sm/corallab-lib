# Third Party
import torch
import einops

import importlib
import os.path

from torch_robotics.environments.primitives import MultiSphereField

# cuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml

from corallab_lib import Pose, RobotPoses
from corallab_lib.visualization import plot_frame
import corallab_assets

from . import robots
from ..robot_interface import RobotInterface


curobo_config_map = {
    "Panda": "franka.yml",
    "UR5": "ur5e_robotiq_2f_140.yml",
    "DualUR10": "dual_ur10e.yml",
    "DualUR5": "dual_ur5/dual_ur5.yml",
    "DualUR10_TEST": "dual_ur10/dual_ur10.py",
}




class CuroboRobot(RobotInterface):

    def __init__(
            self,
            id: str,
            **kwargs
    ):
        self.id = id

        RobotClass = getattr(robots, id)
        self.robot_impl = RobotClass(**kwargs)
        self.config = self.robot_impl.config
        self.retract_config = self.robot_impl.retract_config

    def set_id(self, id):
        self.id = id

    @property
    def robot_id(self):
        return self.robot_impl.robot_id

    @property
    def name(self):
        return self.id

    @property
    def q_dim(self):
        return self.robot_impl.kin_model.kinematics_config.n_dof

    def link_names(self):
        return self.config.kinematics.link_names

    # @property
    # def ws_dim(self):
    #     breakpoint()
    #     return self.robot_impl.kin_model.kinematics_config.n_dof

    def get_position(self, trajs):
        return trajs[..., :self.get_n_dof()]

    def get_velocity(self, trajs):
        return trajs[..., self.get_n_dof():]

    def get_n_dof(self):
        return self.robot_impl.kin_model.kinematics_config.n_dof

    def get_q_min(self):
        return self.robot_impl.kin_model.kinematics_config.joint_limits.position[0]

    def get_q_max(self):
        return self.robot_impl.kin_model.kinematics_config.joint_limits.position[1]

    def get_base_poses(self):
        zeros_q = torch.zeros((1, self.get_n_dof()), **self.tensor_args.as_torch_dict())
        # breakpoint()
        # self.kin_model.get_state(zeros_q)
        return None

    # def random_q(self, n_samples=10):
    #     return self.robot_impl.random_q(n_samples=n_samples)

    # def tmp(self):
    #     # compute forward kinematics:
    #     # torch random sampling might give values out of joint limits
    #     q = torch.rand((10, kin_model.get_dof()), **vars(tensor_args))
    #     out = self.kin_model.get_state(q)

    def fk_map_collision(self, qs, **kwargs):

        if qs.ndim == 3:
            b, h, dof = qs.shape
            qs = qs.view(b * h, dof)
        else:
            b = 1
            h = 1

        kin_state = self.robot_impl.kin_model.get_state(qs)
        spheres = kin_state.link_spheres_tensor.view(b, h, -1, 4)
        return spheres

    def differentiable_fk(self, qs, **kwargs):

        qs = qs.cuda()

        if qs.ndim == 3:
            b, h, dof = qs.shape
            qs = qs.view(b * h, dof)
        else:
            b = 1
            h = 1

        kin_state = self.robot_impl.kin_model.get_state(qs)
        link_poses_tmp = kin_state.link_poses
        link_poses = RobotPoses({
            k: Pose(v.position, v.quaternion) for k, v in link_poses_tmp.items()
        })
        return link_poses

    # Rendering

    def render(self, ax, q=None, color='blue', arrow_length=0.15, arrow_alpha=1.0, arrow_linewidth=2.0,
               draw_links_spheres=True, **kwargs):
        # # draw skeleton
        # skeleton = get_skeleton_from_model(self.diff_ur5, q, self.diff_ur5.get_link_names())
        # skeleton.draw_skeleton(ax=ax, color=color)

        if q.ndim == 1:
            q = q.unsqueeze(0)

        if q.ndim > 2:
            raise NotImplementedError("Rendering states with more than 2 dims is not supported")

        # forward kinematics
        link_poses = self.differentiable_fk(q)

        # draw link collision points
        if draw_links_spheres:
            spheres = self.fk_map_collision(q).squeeze((0, 1))
            spheres_pos = spheres[:, :3]
            spheres_radii = spheres[:, 3]

            # spheres = self.robot_impl.kin_model.get_robot_as_spheres(q)[0]
            # spheres_pos = [s.pose[:3] for s in spheres]
            # spheres_radii = [s.radius for s in spheres]

            sphere_field = MultiSphereField(
                spheres_pos,
                spheres_radii,
            )
            sphere_field.render(ax, color='red', cmap='Reds', **kwargs)

        # draw link frames
        for name, pose in link_poses.items():
            plot_frame(ax, pose)
            # arrow_length=arrow_length, arrow_alpha=arrow_alpha, arrow_linewidth=arrow_linewidth

        # # draw grasped object
        # if self.grasped_object is not None:
        #     frame_grasped_object = fks_dict[self.link_name_grasped_object]

        #     # draw object
        #     pos = frame_grasped_object.translation.squeeze()
        #     ori = q_convert_wxyz(frame_grasped_object.get_quaternion().squeeze())
        #     self.grasped_object.render(ax, pos=pos, ori=ori, color=color)

        #     # draw object collision points
        #     points_in_object_frame = self.grasped_object.base_points_for_collision
        #     points_in_robot_base_frame = frame_grasped_object.transform_point(points_in_object_frame).squeeze()
        #     points_in_robot_base_frame_np = to_numpy(points_in_robot_base_frame)
        #     ax.scatter(
        #         points_in_robot_base_frame_np[:, 0],
        #         points_in_robot_base_frame_np[:, 1],
        #         points_in_robot_base_frame_np[:, 2],
        #         color=color
        #     )
        pass

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            for traj, color in zip(trajs_pos, colors):
                for t in range(traj.shape[0]):
                    q = traj[t]
                    self.render(ax, q, color, **kwargs, arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.)

        if start_state is not None:
            self.render(ax, start_state, color='green')
        if goal_state is not None:
            self.render(ax, goal_state, color='purple')


    # Multi-Agent API

    def is_multi_agent(self):
        return self.robot_impl.is_multi_agent()

    def get_subrobots(self):
        return self.robot_impl.get_subrobots()

    # def separate_joint_state(self, q):
    #     states = []

    #     for i, r in enumerate(self.robot_impl.subrobots):
    #         subrobot_state = r.get_position(joint_state)
    #         states.append(subrobot_state)

    #         joint_state = joint_state[..., r.get_n_dof():]

    #     return states
