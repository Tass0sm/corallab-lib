import torch

from torch_robotics import environments
from torch_robotics.robots import RobotPointMass, RobotPointMass3D
from torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionWorkspaceBoundariesDistanceField,
    CollisionSelfField,
    CollisionObjectDistanceField
)


class TorchRoboticsEnv:
    def __init__(
            self,
            id: str,
            impl = None,
            ws_limits = None,
            tensor_args: dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        if impl:
            self.env_impl = impl
        else:
            EnvClass = getattr(environments, id)
            self.env_impl = EnvClass(
                **kwargs,
                tensor_args=tensor_args
            )

        self.ws_limits = ws_limits or self.env_impl.limits
        self.ws_min = self.ws_limits[0]
        self.ws_max = self.ws_limits[1]

        # FOR SINGLE POINT SDF QUERY:
        # Add distance fields using a point mass robot, because a
        # robot is required for these distance fields.
        if self.get_ws_dim() == 2:
            dummy_robot = RobotPointMass(
                tensor_args=tensor_args
            )
        elif self.get_ws_dim() == 3:
            dummy_robot = RobotPointMass3D(
                tensor_args=tensor_args
            )
        else:
            raise NotImplementedError()

        if self.env_impl.obj_fixed_list is not None:
            self.df_collision_objects = CollisionObjectDistanceField(
                dummy_robot,
                df_obj_list_fn=self.env_impl.get_df_obj_list,
                link_idxs_for_collision_checking=dummy_robot.link_idxs_for_object_collision_checking,
                num_interpolated_points=dummy_robot.num_interpolated_points_for_object_collision_checking,
                link_margins_for_object_collision_checking_tensor=dummy_robot.link_margins_for_object_collision_checking_tensor,
                tensor_args=self.env_impl.tensor_args
            )
        else:
            self.df_collision_objects = None

        self.df_collision_ws_boundaries = CollisionWorkspaceBoundariesDistanceField(
            dummy_robot,
            link_idxs_for_collision_checking=dummy_robot.link_idxs_for_object_collision_checking,
            num_interpolated_points=dummy_robot.num_interpolated_points_for_object_collision_checking,
            link_margins_for_object_collision_checking_tensor=dummy_robot.link_margins_for_object_collision_checking_tensor,
            ws_bounds_min=self.ws_min,
            ws_bounds_max=self.ws_max,
            tensor_args=self.env_impl.tensor_args
        )

    @classmethod
    def from_impl(cls, impl, **kwargs):
        # id is None
        return cls(None, impl=impl, **kwargs)

    @property
    def id(self):
        return self.env_impl.name

    @property
    def name(self):
        return self.env_impl.name

    def get_ws_dim(self):
        return self.env_impl.dim

    def get_limits(self):
        return self.env_impl.limits

    def sample_points(self, n_points, limit_offsets=None):
        limits = self.get_limits()
        if limit_offsets is None:
            limit_offsets = torch.zeros_like(limits)
        limits = limits + limit_offsets

        args = [x for x in limits]
        env_uniform_dist = torch.distributions.uniform.Uniform(*args)
        env_points = env_uniform_dist.sample((n_points,))
        return env_points

    def compute_sdf(self, x):
        # Object collision
        if self.df_collision_objects is not None:
            objects_sdf = self.df_collision_objects.compute_cost(x, x, field_type='sdf')
        else:
            objects_sdf = 0

        # Workspace boundaries
        if self.df_collision_ws_boundaries is not None:
            border_sdf = self.df_collision_ws_boundaries.compute_cost(x, x, field_type="sdf")
        else:
            border_sdf = 0

        return objects_sdf # torch.min(objects_sdf, border_sdf)

    def compute_collision(self, x):
        # Object collision
        if self.df_collision_objects is not None:
            cost_collision_objects = self.df_collision_objects.compute_cost(x, x, field_type='occupancy')
        else:
            cost_collision_objects = 0

        # Workspace boundaries
        if self.df_collision_ws_boundaries is not None:
            cost_collision_border = self.df_collision_ws_boundaries.compute_cost(x, x, field_type="occupancy")
        else:
            cost_collision_border = 0

        collision = cost_collision_objects | cost_collision_border

        return collision
