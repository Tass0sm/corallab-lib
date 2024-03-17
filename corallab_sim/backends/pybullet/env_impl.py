import numpy as np
import pybullet as p
import pybullet_data

from pybullet_utils.bullet_client import BulletClient


class PybulletEnv:
    def __init__(
            self,
            connection_mode=p.GUI,
            ws_limits=None,
            add_plane=True,
            **kwargs
    ):
        self.client = BulletClient(
            connection_mode=connection_mode,
            **kwargs
        )
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0, 0, -9.81)

        self.ws_limits = ws_limits or np.array([[-1, -1, 0],
                                                [ 1,  1, 1]])
        self.ws_min = self.ws_limits[0]
        self.ws_max = self.ws_limits[1]

        if add_plane:
            plane_height = self.ws_min[2]
            self.plane_id = self.client.loadURDF("plane.urdf", basePosition=(0, 0, plane_height))
        else:
            self.plane_id = None        

    def get_ws_dim(self):
        return 3

    def get_limits(self):
        return self.ws_limits

    def sample_points(self, n_points, limit_offsets=None):
        limits = self.get_limits()
        if limit_offsets is None:
            limit_offsets = torch.zeros_like(limits)
        limits = limits + limit_offsets

        args = [x for x in limits]
        env_uniform_dist = torch.distributions.uniform.Uniform(*args)
        env_points = env_uniform_dist.sample((n_points,))
        return env_points

    def compute_collision(self, x):
        pass
