import numpy as np
import torch
import einops
from itertools import product

from corallab_lib import Env, Robot

# from torch_robotics.tasks.dynamic_planning_task import DynamicPlanningTask
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch, to_numpy
# from torch_robotics.environments.primitives import MultiSphereField
# from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionObjectDistanceField

from curobo.geom.types import WorldConfig, Sphere
from curobo.geom.sdf.world import WorldCollisionConfig, WorldPrimitiveCollision

import scipy.interpolate

from .env_impl import CuroboEnv
from .robot_impl import CuroboRobot

from .motion_planning_problem_impl import CuroboMotionPlanningProblem

from scipy.spatial.distance import cdist


class DynamicObstacle:

    def __init__(
            self,
            entity,
            traj,
            timesteps,
            tensor_args : dict = {}
    ):
        if isinstance(traj, np.ndarray):
            traj = torch.tensor(traj, **tensor_args)

        if traj.ndim == 2:
            traj = traj.unsqueeze(0)

        assert timesteps.shape[0] == traj.shape[1]

        self.entity = entity
        self.traj = traj
        self.timesteps = timesteps.to(**tensor_args)
        self.tensor_args = tensor_args

    def dynamic_state(
            self,
            time : torch.Tensor
    ):
        time = time.clamp(self.timesteps[0], self.timesteps[-1])

        timesteps_cpu = self.timesteps.cpu()
        traj_cpu = self.traj.cpu().squeeze()
        interpolator = scipy.interpolate.interp1d(
            timesteps_cpu,
            traj_cpu,
            axis=0, bounds_error=False, fill_value=(traj_cpu[0], traj_cpu[-1])
        )

        traj_at_time = interpolator(time.cpu())
        traj_at_time = to_torch(traj_at_time, **self.tensor_args)

        if traj_at_time.nelement() == 0:
            raise NotImplementedError

        return traj_at_time.unsqueeze(0)

    def dynamic_fk(
            self,
            time : int,
            **kwargs
    ):
        q = self.dynamic_state(time)
        return self.entity.fk_map_collision(q, **kwargs)

    # def get_margins(self):
    #     margins = to_torch(self.entity.link_margins_for_object_collision_checking, **self.tensor_args)
    #     return margins


class CuroboDynamicPlanningProblem(CuroboMotionPlanningProblem):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(None, **kwargs)

        self.dynamic_obstacles = []

    def add_dynamic_obstacle(self, robot, traj, time):
        obstacle = DynamicObstacle(robot, traj, time, tensor_args=self.tensor_args)
        self.dynamic_obstacles.append(obstacle)

    def clear_dynamic_obstacles(self):
        self.dynamic_obstacles = []

    def static_compute_collision(self, qs, **kwargs):
        return super().compute_collision(qs, **kwargs)

    def compute_collision(self, time, qs, **kwargs):
        qs = to_torch(qs, **self.tensor_args)

        b, h, dof = qs.shape
        qs = qs.view(b * h, dof)

        time = to_torch(time, **self.tensor_args)

        in_static_collision = super().compute_collision(qs, **kwargs)
        in_static_collision = in_static_collision.view(b, h)

        if len(self.dynamic_obstacles) > 0:
            dynamic_spheres = self._get_dynamic_spheres(time)
            dynamic_spheres_np = dynamic_spheres.cpu().numpy()

            # run collision, self collision, bounds
            kin_state = self.curobo_fn.get_kinematics(qs)
            main_spheres = kin_state.link_spheres_tensor.view(b, h, -1, 4)
            main_spheres_np = main_spheres.cpu().numpy()

            assert dynamic_spheres_np.shape[0] == main_spheres_np.shape[0]
            assert dynamic_spheres_np.shape[1] == main_spheres_np.shape[1]
            n_b = main_spheres_np.shape[0]
            n_h = main_spheres_np.shape[1]

            max_overlaps_h_l = []

            for b in range(n_b):
                dynamic_spheres_traj = dynamic_spheres_np[b]
                main_spheres_traj = main_spheres_np[b]

                max_overlap_i_l = []

                for i in range(n_h):
                    dynamic_spheres_i = dynamic_spheres_traj[i]
                    main_spheres_i = main_spheres_traj[i]

                    dynamic_centers_i = dynamic_spheres_i[:, :3]
                    dynamic_radii_i = dynamic_spheres_i[:, 3:]
                    main_centers_i = main_spheres_i[:, :3]
                    main_radii_i = main_spheres_i[:, 3:]

                    dist_matrix = cdist(dynamic_centers_i, main_centers_i)
                    buffer_matrix = cdist(dynamic_radii_i, main_radii_i, lambda x, y: x + y)
                    overlap_matrix = buffer_matrix - dist_matrix

                    max_overlap_i = overlap_matrix.max()
                    max_overlap_i_l.append(max_overlap_i)

                max_overlaps_h = torch.tensor(max_overlap_i_l)
                max_overlaps_h_l.append(max_overlaps_h)

            max_overlaps = torch.stack(max_overlaps_h_l)

            in_dynamic_collision = (max_overlaps > 0).to(device=self.tensor_args["device"])
            in_collision = in_static_collision | in_dynamic_collision
        else:
            in_collision = in_static_collision

        return in_collision

    def _get_dynamic_spheres(self, time):
        """
        returns
        - spheres: batch x time x 4
        """

        spheres_l = []

        for obstacle in self.dynamic_obstacles:
            spheres_i = obstacle.dynamic_fk(time)
            if spheres_i is None:
                continue

            spheres_l.append(spheres_i)

        spheres = torch.concatenate(spheres_l, dim=2)

        return spheres


    # def random_coll_free_q(self, *args, **kwargs):
    #     return self.task_impl.random_coll_free_q(*args, **kwargs)

    # # def compute_collision_cost(self, trajs):
    # #     pass
