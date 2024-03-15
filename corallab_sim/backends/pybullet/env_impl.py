import pybullet as p
from pybullet_utils.bullet_client import BulletClient


class PyBulletEnv:
    def __init__(
            self,
            connection_mode=p.GUI,
            ws_limits = None,
            **kwargs
    ):
        self.client = BulletClient(
            connection_mode=connection_mode,
            **kwargs
        )

        self.ws_limits = ws_limits or torch.tensor([[-1, -1, -1],
                                                    [ 1,  1,  1]])
        self.ws_min = self.ws_limits[0]
        self.ws_max = self.ws_limits[1]

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
