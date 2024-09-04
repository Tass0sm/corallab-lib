import numpy as np


class EnvFloor3D:

    @classmethod
    def setup(cls, env):
        env.ws_limits = np.array([[-1, -1, -0.1],
                                  [ 1,  1, 1]])
        env.ws_min = env.ws_limits[0]
        env.ws_max = env.ws_limits[1]

        plane_height = env.ws_min[2]
        env.plane_id = env.client.loadURDF("plane.urdf", basePosition=(0, 0, plane_height))
        env.pb_objs.append(env.plane_id)
