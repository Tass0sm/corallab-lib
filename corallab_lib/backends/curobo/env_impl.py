# Third Party
import torch

# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


class CuroboEnv:
    def __init__(
            self,
            id: str,
            # impl = None,
            # ws_limits = None,
            # tensor_args: dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):

        tensor_args = TensorDeviceType()

        self.config_file = None

        # create a world from a dictionary of objects
        # cuboid: {} # dictionary of objects that are cuboids
        # mesh: {} # dictionary of objects that are meshes
        self.config = {
            "cuboid": {
                "table": {"dims": [2, 2, 0.2], "pose": [0.4, 0.0, -0.1, 1, 0, 0, 0]},
                "cube_1": {"dims": [0.1, 0.1, 0.2], "pose": [0.4, 0.0, 0.5, 1, 0, 0, 0]},
            },
            "mesh": {
                "scene": {
                    "pose": [1.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
                    "file_path": "scene/nvblox/srl_ur10_bins.obj",
                }
            },
        }

        # if impl:
        #     self.env_impl = impl
        # else:
        #     EnvClass = getattr(environments, id)
        #     self.env_impl = EnvClass(
        #         **kwargs,
        #         tensor_args=tensor_args
        #     )

        # self.ws_limits = ws_limits or self.env_impl.limits
        # self.ws_min = self.ws_limits[0]
        # self.ws_max = self.ws_limits[1]

        # # FOR SINGLE POINT SDF QUERY:
        # # Add distance fields using a point mass robot, because a
        # # robot is required for these distance fields.
        # if self.get_ws_dim() == 2:
        #     dummy_robot = RobotPointMass(
        #         tensor_args=tensor_args
        #     )
        # elif self.get_ws_dim() == 3:
        #     dummy_robot = RobotPointMass3D(
        #         tensor_args=tensor_args
        #     )
        # else:
        #     raise NotImplementedError()



    # @classmethod
    # def from_impl(cls, impl, **kwargs):
    #     # id is None
    #     return cls(None, impl=impl, **kwargs)

    # @property
    # def name(self):
    #     return self.env_impl.name

    # def get_ws_dim(self):
    #     return self.env_impl.dim

    # def get_limits(self):
    #     return self.env_impl.limits

    # def sample_points(self, n_points, limit_offsets=None):
    #     limits = self.get_limits()
    #     if limit_offsets is None:
    #         limit_offsets = torch.zeros_like(limits)
    #     limits = limits + limit_offsets

    #     args = [x for x in limits]
    #     env_uniform_dist = torch.distributions.uniform.Uniform(*args)
    #     env_points = env_uniform_dist.sample((n_points,))
    #     return env_points

    # def compute_sdf(self, x):
    #     # Object collision
    #     if self.df_collision_objects is not None:
    #         objects_sdf = self.df_collision_objects.compute_cost(x, x, field_type='sdf')
    #     else:
    #         objects_sdf = 0

    #     # Workspace boundaries
    #     if self.df_collision_ws_boundaries is not None:
    #         border_sdf = self.df_collision_ws_boundaries.compute_cost(x, x, field_type="sdf")
    #     else:
    #         border_sdf = 0

    #     return objects_sdf # torch.min(objects_sdf, border_sdf)

    # def compute_collision(self, x):
    #     # Object collision
    #     if self.df_collision_objects is not None:
    #         cost_collision_objects = self.df_collision_objects.compute_cost(x, x, field_type='occupancy')
    #     else:
    #         cost_collision_objects = 0

    #     # Workspace boundaries
    #     if self.df_collision_ws_boundaries is not None:
    #         cost_collision_border = self.df_collision_ws_boundaries.compute_cost(x, x, field_type="occupancy")
    #     else:
    #         cost_collision_border = 0

    #     collision = cost_collision_objects | cost_collision_border

    #     return collision
