import numpy as np
import torch

from .env_impl import CuroboEnv
from .robot_impl import CuroboRobot

from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.util.trajectory import *

class CuroboTask:
    def __init__(
            self,
            id : str,
            env = None,
            robot = None,
            **kwargs
    ):
        assert isinstance(env, CuroboEnv) or env is None
        assert isinstance(robot, CuroboRobot) or robot is None

        self.env = env
        self.robot = robot

        config = RobotWorldConfig.load_from_config(
            robot.config,
            env.config,
            collision_activation_distance=0.0
        )
        self.curobo_fn = RobotWorld(config)

    def get_q_dim(self):
        return self.robot.get_n_dof()

    def get_q_min(self):
        return self.robot.get_q_min()

    def get_q_max(self):
        return self.robot.get_q_max()

    def random_coll_free_q(self, n_samples=1, max_samples=1000, max_tries=1000):
        # Random position in configuration space not in collision
        reject = True
        samples = torch.zeros((n_samples, self.robot.get_n_dof()), **self.curobo_fn.tensor_args.as_torch_dict())
        idx_begin = 0

        for i in range(max_tries):
            free_qs = self.curobo_fn.sample(n_samples, mask_valid=True)
            idx_end = min(idx_begin + free_qs.shape[0], samples.shape[0])
            samples[idx_begin:idx_end] = free_qs[:idx_end - idx_begin]
            idx_begin = idx_end

            if idx_end >= n_samples:
                reject = False
                break

        if reject:
            sys.exit("Could not find a collision free configuration")

        return samples.squeeze(), None

    def compute_collision(self, qs, margin=0.0):
        if isinstance(qs, np.ndarray):
            qs = torch.tensor(qs, **self.curobo_fn.tensor_args.as_torch_dict())

        if qs.ndim == 1:
            qs = qs.unsqueeze(0)

        return self.curobo_fn.validate(qs).logical_not()

    def compute_collision_cost(self, trajs):
        pass

    def get_trajs_collision_and_free(self, trajs, return_indices=False, num_interpolation=5):
        assert trajs.ndim == 3 or trajs.ndim == 4
        N = 1
        if trajs.ndim == 4:  # n_goals (or steps), batch of trajectories, length, dim
            N, B, H, D = trajs.shape
            trajs_new = einops.rearrange(trajs, 'N B H D -> (N B) H D')
        else:
            B, H, D = trajs.shape
            trajs_new = trajs

        ###############################################################################################################
        # compute collisions on a finer interpolated trajectory

        # TODO: Make interpolation?
        trajs_interpolated = trajs_new
        # trajs_interpolated = interpolate_traj_via_points(trajs_new, num_interpolation=num_interpolation)

        # Set 0 margin for collision checking, which means we allow trajectories to pass very close to objects.
        # While the optimized trajectory via points are not at a 0 margin from the object, the interpolated via points
        # might be. A 0 margin guarantees that we do not discard those trajectories, while ensuring they are not in
        # collision (margin < 0).

        # , debug=True
        trajs_waypoints_valid = self.curobo_fn.validate_trajectory(trajs_interpolated)

        if trajs.ndim == 4:
            trajs_waypoints_collisions = einops.rearrange(trajs_waypoints_collisions, '(N B) H -> N B H', N=N, B=B)

        trajs_free_idxs = torch.argwhere(trajs_waypoints_valid.all(dim=-1))
        trajs_coll_idxs = torch.argwhere(trajs_waypoints_valid.logical_not().any(dim=-1))

        ###############################################################################################################
        # Check that trajectories that are not in collision are inside the joint limits
        if trajs_free_idxs.nelement() == 0:
            pass
        else:
            if trajs.ndim == 4:
                trajs_free_tmp = trajs[trajs_free_idxs[:, 0], trajs_free_idxs[:, 1], ...]
            else:
                trajs_free_tmp = trajs[trajs_free_idxs.squeeze(), ...]

            trajs_free_tmp_position = trajs_free_tmp # self.robot.get_position(trajs_free_tmp)

            # if self.robot.name == "MultiRobot":
            #     trajs_free_tmp_position = self.robot.safe_select_free_q(trajs_free_tmp_position)

            trajs_free_inside_joint_limits_idxs = torch.logical_and(
                trajs_free_tmp_position >= self.get_q_min(),
                trajs_free_tmp_position <= self.get_q_max()
            ).all(dim=-1).all(dim=-1)
            trajs_free_inside_joint_limits_idxs = torch.atleast_1d(trajs_free_inside_joint_limits_idxs)
            trajs_free_idxs_try = trajs_free_idxs[torch.argwhere(trajs_free_inside_joint_limits_idxs).squeeze()]
            if trajs_free_idxs_try.nelement() == 0:
                trajs_coll_idxs = trajs_free_idxs.clone()
            else:
                trajs_coll_idxs_joint_limits = trajs_free_idxs[torch.argwhere(torch.logical_not(trajs_free_inside_joint_limits_idxs)).squeeze()]
                if trajs_coll_idxs_joint_limits.ndim == 1:
                    trajs_coll_idxs_joint_limits = trajs_coll_idxs_joint_limits[..., None]
                trajs_coll_idxs = torch.cat((trajs_coll_idxs, trajs_coll_idxs_joint_limits))
            trajs_free_idxs = trajs_free_idxs_try

        ###############################################################################################################
        # Return trajectories free and in collision
        if trajs.ndim == 4:
            trajs_free = trajs[trajs_free_idxs[:, 0], trajs_free_idxs[:, 1], ...]
            if trajs_free.ndim == 2:
                trajs_free = trajs_free.unsqueeze(0).unsqueeze(0)
            trajs_coll = trajs[trajs_coll_idxs[:, 0], trajs_coll_idxs[:, 1], ...]
            if trajs_coll.ndim == 2:
                trajs_coll = trajs_coll.unsqueeze(0).unsqueeze(0)
        else:
            trajs_free = trajs[trajs_free_idxs.squeeze(), ...]
            if trajs_free.ndim == 2:
                trajs_free = trajs_free.unsqueeze(0)
            trajs_coll = trajs[trajs_coll_idxs.squeeze(), ...]
            if trajs_coll.ndim == 2:
                trajs_coll = trajs_coll.unsqueeze(0)

        if trajs_coll.nelement() == 0:
            trajs_coll = None
        if trajs_free.nelement() == 0:
            trajs_free = None

        if return_indices:
            return trajs_coll, trajs_coll_idxs, trajs_free, trajs_free_idxs, trajs_waypoints_valid.logical_not()

        return trajs_coll, trajs_free

    def compute_fraction_free_trajs(self, trajs, **kwargs):
        return self.curobo_fn.validate_trajectory(trajs).all(axis=-1).float().mean()

    def compute_collision_intensity_trajs(self, trajs, **kwargs):
        collision_at_points = self.curobo_fn.validate_trajectory(trajs).logical_not().float()
        return torch.count_nonzero(collision_at_points) / collision_at_points.nelement()

    def compute_success_free_trajs(self, trajs, **kwargs):
        # If at least one trajectory is collision free, then we consider success
        if self.curobo_fn.validate_trajectory(trajs).all(axis=-1).any():
            return 1
        else:
            return 0
