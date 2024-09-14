import sys
import numpy as np
import torch
from jaxtyping import Array, Float, Bool

from torch_robotics.torch_utils.torch_utils import to_torch, to_numpy
from .env_impl import CuroboEnv
from .robot_impl import CuroboRobot

from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig, sum_mask, mask
from curobo.util.sample_lib import HaltonGenerator
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.util.trajectory import *


class CuroboMotionPlanningProblem:
    def __init__(
            self,
            env = None,
            robot = None,
            self_collision_activation_distance: float = 0.05,
            val_self_collision_activation_distance: float = 0.0,
            val_world_collision_activation_distance: float = 0.0,
            local_collision_step : float = 0.03,
            local_collision_max_dist : float = 0.1,
            seed : int = 0,
            **kwargs
    ):
        assert isinstance(env, CuroboEnv) or env is None
        assert isinstance(robot, CuroboRobot) or robot is None

        self.env = env
        self.robot = robot

        self.local_collision_step = local_collision_step
        self.local_collision_max_dist = local_collision_max_dist

        config = RobotWorldConfig.load_from_config(
            robot.config,
            env.config,
            # collision_activation_distance=collision_activation_distance,
            # self_collision_activation_distance=self_collision_activation_distance,
        )

        sample_gen_override = HaltonGenerator(
            config.kinematics.get_dof(),
            config.tensor_args,
            up_bounds=config.kinematics.get_joint_limits().position[1],
            low_bounds=config.kinematics.get_joint_limits().position[0],
            seed=seed,
        )
        config.sampler = sample_gen_override

        self.curobo_fn = RobotWorld(config)

        # FOR COLLISION_FREE INVERSE KINEMATICS

        # ik_config = IKSolverConfig.load_from_robot_config(
        #     robot.config,
        #     env.config,
        #     rotation_threshold=0.05,
        #     position_threshold=0.005,
        #     num_seeds=20,
        #     self_collision_check=True,
        #     self_collision_opt=True,
        #     # use_gradient_descent=True,
        #     # store_debug=True,
        #     tensor_args=self.curobo_fn.tensor_args,
        #     use_cuda_graph=True,
        # )
        # self.ik_solver = IKSolver(ik_config)

        self.tensor_args = self.curobo_fn.tensor_args.as_torch_dict()

        self.self_collision_activation_distance = self_collision_activation_distance
        self.val_self_collision_activation_distance = val_self_collision_activation_distance
        self.val_world_collision_activation_distance = val_world_collision_activation_distance

    def get_q_dim(self):
        return self.robot.get_n_dof()

    def get_q_min(self):
        return self.robot.get_q_min()

    def get_q_max(self):
        return self.robot.get_q_max()

    def random_q(self, n_samples=1, **kwargs):
        return self.curobo_fn.sample(n_samples, mask_valid=False, **kwargs)

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

    # def coll_free_ik(
    #         self,
    #         link_poses,
    #         num_solutions : int = 1,
    #         current_config = None
    # ):
    #     """Poses is a dict from link name to Pose. Eventually this will be it's
    #     own object.

    #     """

    #     batch_size = link_poses["ee_link_0"].shape[0]

    #     if current_config is not None:
    #         seed_config = current_config.unsqueeze(0).unsqueeze(1).expand(batch_size, -1, -1)
    #         # retract_conifg = current_config.unsqueeze(0).expand(batch_size, -1)
    #     else:
    #         seed_config = None
    #         # retract_conifg = None

    #     ik_results = self.ik_solver.solve_batch(
    #         link_poses["ee_link_0"],
    #         link_poses=link_poses,
    #         return_seeds=num_solutions,
    #         seed_config=seed_config,
    #         # retract_config=retract_conifg,
    #     )

    #     return ik_results # .js_solution.position

    def distance_q(self, q1, q2):
        return torch.linalg.norm(q1 - q2, dim=-1)

    def local_motion(self, q1, q2, step=None, no_max_dist=False):
        """
        TODO: Make it depend on task
        max_step=0.08, max_dist=0.1
        """

        step = step or self.local_collision_step
        max_dist = None if no_max_dist else self.local_collision_max_dist

        q1 = to_torch(q1, **self.tensor_args)
        q2 = to_torch(q2, **self.tensor_args)

        dist = self.distance_q(q1, q2)
        # print(f"Local motion with dist: {dist}")
        if max_dist is not None and dist > max_dist:
            q2 = q1 + (q2 - q1) * (max_dist / dist)
            dist = max_dist

        alpha = torch.linspace(0, 1, int(dist / step) + 2, **self.tensor_args)
        alpha = alpha.unsqueeze(1)

        q1 = q1.unsqueeze(0)
        q2 = q2.unsqueeze(0)
        extension = q1 + (q2 - q1) * alpha
        return extension.squeeze()

    def check_local_motion(self, q1, q2, step=None, no_max_dist=False, **kwargs):
        local_motion_states = self.local_motion(q1, q2, step=step, no_max_dist=no_max_dist)
        any_collision = self.compute_collision(local_motion_states, **kwargs).any().item()
        return not any_collision

    def check_collision(self, q: Float[Array, "b h d"], **kwargs) -> Bool[Array, "b h"]:
        if isinstance(q, np.ndarray):
            q = torch.tensor(q, **self.curobo_fn.tensor_args.as_torch_dict())

        b, h, dof = q.shape
        flat_qs = q.reshape((b*h, dof))
        colls = self.curobo_fn.validate(flat_qs).logical_not()
        return colls.reshape((b, h))

    def compute_collision_info(self, qs, margin=0.0):
        info = {
            "cost_collision_objects": 0.0,
            "cost_collision_border": 0.0,
            "self_collision_robots": None,
        }

        in_collision = self.check_collision(qs, margin=margin)

        # TODO: FIX THIS TERRIBLE HACK
        if in_collision:
            info["self_collision_robots"] = torch.tensor([[0, 0, 1]])

        return in_collision, info

    def compute_collision_cost(
            self,
            trajs,
            env_query_idx: Optional[torch.Tensor] = None,
            self_margin_override : Optional[float] = None,
    ):
        """
        trajs: batch , horizon, dof
        env_query_idx: batch, 1
        """

        # b, h, dof = trajs.shape
        # q = trajs.view(b * h, dof)
        # kin_state = self.curobo_fn.get_kinematics(q)
        # spheres = kin_state.link_spheres_tensor.view(b, h, -1, 4)

        # There is no way to tweak the activation distance for the self
        # collision constraint, so one must add to the radii of the spheres
        # passed to it.

        # self_margin = self_margin_override or self.self_collision_activation_distance
        # spheres_for_self_coll = self._get_inflated_spheres(spheres, self_margin)
        # d_self = self.curobo_fn.get_self_collision(spheres_for_self_coll)
        # d_world = self.curobo_fn.get_collision_constraint(spheres, env_query_idx)
        # d_bound = self.curobo_fn.get_bound(q.view(b, h, dof))

        # cost = d_self + d_world # + d_bound
        # cost = cost.reshape((b, h))
        # cost = cost.sum(-1)

        b, h, q = trajs.shape
        qs = einops.rearrange(trajs, "b h q -> (b h) q")
        kin_state = robot_world.get_kinematics(qs)

        spheres = kin_state.link_spheres_tensor.unsqueeze(1)

        self_margin = 0.1
        spheres_for_self_coll = ctask._get_inflated_spheres(spheres, self_margin)
        d_self = robot_world.self_collision_cost(spheres_for_self_coll)

        d_sdf = robot_world.collision_constraint(spheres)
        cost = d_sdf + d_self
        cost = cost.reshape((b, h))
        cost = cost.sum(-1)

        return cost

    # def get_self_collision_info(self, qs):
    #     if isinstance(qs, np.ndarray):
    #         qs = torch.tensor(qs, **self.curobo_fn.tensor_args.as_torch_dict())

    #     if qs.ndim == 1:
    #         qs = qs.unsqueeze(0)

    #     b, dof = qs.shape
    #     # qs = einops.rearrange(trajs, "b h q -> (b h) q")
    #     kin_state = self.curobo_fn.get_kinematics(qs)

    #     spheres = kin_state.link_spheres_tensor.unsqueeze(1)

    #     self_margin = 0.0
    #     spheres_for_self_coll = self._get_inflated_spheres(spheres, self_margin)

    #     breakpoint()

    #     d_self = self.curobo_fn.self_collision_cost(spheres_for_self_coll)



    #     return {}


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
        trajs_interpolated = trajs_new.contiguous()
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

    def compute_fraction_free_trajs(
            self,
            trajs,
            self_margin_override : Optional[float] = None,
            world_margin_override : Optional[float] = None,
            **kwargs
    ):
        trajs = trajs.contiguous()

        return self._validate_trajectory(
            trajs,
            self_margin_override=self_margin_override,
            world_margin_override=world_margin_override,
        ).all(axis=-1).float().mean()

    def compute_collision_intensity_trajs(
            self,
            trajs,
            self_margin_override : Optional[float] = None,
            world_margin_override : Optional[float] = None,
            **kwargs
    ):
        trajs = trajs.contiguous()

        collision_at_points = self._validate_trajectory(
            trajs,
            self_margin_override=self_margin_override,
            world_margin_override=world_margin_override,
        ).logical_not().float()
        return torch.count_nonzero(collision_at_points) / collision_at_points.nelement()

    def compute_success_free_trajs(
            self,
            trajs,
            self_margin_override : Optional[float] = None,
            world_margin_override : Optional[float] = None,
            **kwargs
    ):
        trajs = trajs.contiguous()

        # If at least one trajectory is collision free, then we consider success
        any_success = self._validate_trajectory(
            trajs,
            self_margin_override=self_margin_override,
            world_margin_override=world_margin_override,
        ).all(axis=-1).any()

        if any_success:
            return 1
        else:
            return 0

    ################################################################################
    # Reimplementations
    ################################################################################

    # def _validate(self, q: torch.Tensor, env_query_idx: Optional[torch.Tensor] = None):
    #     """
    #     This does not support batched environments, use validate_trajectory instead
    #     """
    #     # run collision, self collision, bounds
    #     b, dof = q.shape
    #     kin_state = self.get_kinematics(q)
    #     spheres = kin_state.link_spheres_tensor.view(b, 1, -1, 4)

    #     breakpoint()

    #     d_self = self.get_self_collision(spheres)
    #     d_world = self.get_collision_constraint(spheres, env_query_idx)
    #     d_bound = self.get_bound(q.view(b, 1, dof))
    #     d_mask = sum_mask(d_self, d_world, d_bound)
    #     return d_mask

    def _validate_trajectory(
            self,
            q: torch.Tensor,
            env_query_idx: Optional[torch.Tensor] = None,
            self_margin_override : Optional[float] = None,
            world_margin_override : Optional[float] = None,
    ):
        """
        q: batch , horizon, dof
        env_query_idx: batch, 1
        """
        # run collision, self collision, bounds
        b, h, dof = q.shape
        q = q.view(b * h, dof)
        kin_state = self.curobo_fn.get_kinematics(q)
        spheres = kin_state.link_spheres_tensor.view(b, h, -1, 4)

        # There is no way to tweak the activation distance for the self
        # collision constraint, so one must add to the radii of the spheres
        # passed to it.

        self_margin = self_margin_override or self.val_self_collision_activation_distance
        spheres_for_self_coll = self._get_inflated_spheres(spheres, self_margin)
        d_self = self.curobo_fn.get_self_collision(spheres_for_self_coll)

        world_margin = world_margin_override or self.val_world_collision_activation_distance
        spheres_for_world_coll = self._get_inflated_spheres(spheres, world_margin)
        d_world = self.curobo_fn.get_collision_constraint(spheres_for_world_coll, env_query_idx)

        d_bound = self.curobo_fn.get_bound(q.view(b, h, dof))
        d_mask = mask(d_self, d_world, d_bound)
        return d_mask

    def _get_inflated_spheres(
            self,
            spheres,
            margin : float = 0.0
    ) -> torch.Tensor:
        inflated_spheres = spheres.clone()
        inflated_spheres[..., 3] += margin / 2
        return inflated_spheres
