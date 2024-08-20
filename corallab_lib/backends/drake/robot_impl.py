# Third Party
import torch
import einops

import importlib
import os.path

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

import corallab_assets

from . import robots
from ..robot_interface import RobotInterface


class DrakeRobot():

    def __init__(
            self,
            id: str,
            **kwargs
    ):
        self.id = id

        RobotClass = getattr(robots, id)
        self.robot_impl = RobotClass(**kwargs)

        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
        Parser(plant).AddModels(self.robot_impl.urdf_path)

        world = plant.world_frame()
        base = plant.GetFrameByName("base_fixture_link")
        plant.WeldFrames(world, base)

        plant.Finalize()
        self.plant = plant

    # @property
    # def robot_id(self):
    #     return self.robot_impl.robot_id

    # @property
    # def name(self):
    #     return self.id

    # @property
    # def q_dim(self):
    #     return self.robot_impl.kin_model.kinematics_config.n_dof

    # @property
    # def ws_dim(self):
    #     breakpoint()
    #     return self.robot_impl.kin_model.kinematics_config.n_dof

    # def get_position(self, trajs):
    #     return trajs[..., :self.get_n_dof()]

    # def get_velocity(self, trajs):
    #     return trajs[..., self.get_n_dof():]

    # def get_n_dof(self):
    #     return self.robot_impl.kin_model.kinematics_config.n_dof

    # def get_q_min(self):
    #     return self.robot_impl.kin_model.kinematics_config.joint_limits.position[0]

    # def get_q_max(self):
    #     return self.robot_impl.kin_model.kinematics_config.joint_limits.position[1]

    # def get_base_poses(self):
    #     zeros_q = torch.zeros((1, self.get_n_dof()), **self.tensor_args.as_torch_dict())
    #     # breakpoint()
    #     # self.kin_model.get_state(zeros_q)
    #     return None

    # def random_q(self, n_samples=10):
    #     return self.robot_impl.random_q(n_samples=n_samples)

    # def tmp(self):
    #     # compute forward kinematics:
    #     # torch random sampling might give values out of joint limits
    #     q = torch.rand((10, kin_model.get_dof()), **vars(tensor_args))
    #     out = self.kin_model.get_state(q)

    # def fk_map_collision(self, qs, **kwargs):

    #     if qs.ndim == 3:
    #         b, h, dof = qs.shape
    #         qs = qs.view(b * h, dof)
    #     else:
    #         b = 1
    #         h = 1

    #     kin_state = self.robot_impl.kin_model.get_state(qs)
    #     spheres = kin_state.link_spheres_tensor.view(b, h, -1, 4)
    #     return spheres

    # Multi-Agent API

    # def is_multi_agent(self):
    #     return self.robot_impl.is_multi_agent()

    # def get_subrobots(self):
    #     return self.robot_impl.get_subrobots()

    # def separate_joint_state(self, q):
    #     states = []

    #     for i, r in enumerate(self.robot_impl.subrobots):
    #         subrobot_state = r.get_position(joint_state)
    #         states.append(subrobot_state)

    #         joint_state = joint_state[..., r.get_n_dof():]

    #     return states
