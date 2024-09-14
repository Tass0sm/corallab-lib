import numpy as np

import corallab_assets


class EnvWideTable3D:

    @classmethod
    def setup(cls, env):
        env.ws_limits = np.array([[-1, -1, -0.1],
                                  [ 1,  1, 1]])
        env.ws_min = env.ws_limits[0]
        env.ws_max = env.ws_limits[1]

        # ADD PLANE
        plane_height = env.ws_min[2]
        env.plane_id = env.client.loadURDF("plane.urdf", basePosition=(0, 0, plane_height))
        env.pb_objs.append(env.plane_id)

        # TABLE PARAMETERS
        env.table_center = np.array([0.0, 0.0, 0.0])
        env.table_pose = [*env.table_center.tolist(), 1., 0., 0., 0.]
        env.table_dimensions = np.array([0.3, 1.6, 0.2])
        table_half_dimensions = env.table_dimensions / 2

        env.table_bounds = np.vstack([env.table_center - table_half_dimensions,
                                      env.table_center + table_half_dimensions])
        env.table_height = env.table_bounds[1, 2]

        # ADD TABLE
        table_urdf_path = str(corallab_assets.get_resource_path("objects/wide_table.urdf"))
        env.table_id = env.client.loadURDF(table_urdf_path, basePosition=env.table_center, useFixedBase=True)
        env.pb_objs.append(env.table_id)
