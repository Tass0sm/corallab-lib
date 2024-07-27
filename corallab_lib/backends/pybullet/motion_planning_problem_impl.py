import torch
import numpy as np
import corallab_lib.backends.pybullet.ompl.utils as pb_utils

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch, to_numpy

from itertools import product

from .env_impl import PybulletEnv
from .robot_impl import PybulletRobot


class PybulletMotionPlanningProblem:
    def __init__(
            self,
            env=None,
            robot=None,
            **kwargs
    ):
        assert isinstance(env, PybulletEnv)
        assert isinstance(robot, PybulletRobot)

        self.env = env
        self.robot = robot

        self.gen = np.random.default_rng(seed=0)

        self.robot.robot_impl.load(
            urdf_override=self.robot.urdf_override
        )

        self._setup_collision_detection(self.robot, self.env)

    def _setup_collision_detection(self, robot, env, allow_collision_links=[]):
        pybullet_robots = [self.robot.robot_impl]

        self.intra_robot_link_pairs = []
        for pybullet_bot in pybullet_robots:
            pairs = pb_utils.get_self_link_pairs(
                pybullet_bot.id,
                pybullet_bot.arm_controllable_joints
            )

            self.intra_robot_link_pairs.extend(pairs)

        moving_bodies = []
        all_moving_links = []
        self.inter_robot_link_pairs = []
        for pybullet_bot in pybullet_robots:
            bot_moving_links = pb_utils.get_moving_links(
                pybullet_bot.id,
                pybullet_bot.arm_controllable_joints
            )

            full_bot_moving_links = [pb_utils.Link(pybullet_bot.id, x) for x in bot_moving_links]
            self.inter_robot_link_pairs.extend(product(all_moving_links, full_bot_moving_links))
            all_moving_links.extend(full_bot_moving_links)

            moving_bodies.append((pybullet_bot.id, bot_moving_links))

        self.robot_object_body_pairs = list(product(moving_bodies, self.env.pb_objs))

    def get_q_dim(self):
        return self.robot.get_q_dim()

    def get_q_min(self):
        return self.robot.get_q_min()

    def get_q_max(self):
        return self.robot.get_q_max()

    def random_coll_free_q(self, n_samples=1, max_samples=100, max_tries=1000):
        # Random position in configuration space not in collision
        reject = True
        samples = np.zeros((n_samples, self.robot.robot_impl.arm_num_dofs))
        idx_begin = 0

        for i in range(max_tries):
            qs = self.robot.random_q(self.gen, max_samples)

            in_collision = self.compute_collision(qs)
            idxs_not_in_collision = np.argwhere(in_collision == False)
            if idxs_not_in_collision.size == 0:
                # all points are in collision
                continue

            idxs_not_in_collision = idxs_not_in_collision.squeeze(axis=1)
            idx_random = self.gen.permutation(len(idxs_not_in_collision))[:n_samples]
            free_qs = qs[idxs_not_in_collision[idx_random]]
            idx_end = min(idx_begin + free_qs.shape[0], samples.shape[0])
            samples[idx_begin:idx_end] = free_qs[:idx_end - idx_begin]
            idx_begin = idx_end
            if idx_end >= n_samples:
                reject = False
                break

        if reject:
            sys.exit("Could not find a collision free configuration")

        return samples.squeeze(), None


    def compute_collision(self, qs, **kwargs):

        # Helper
        def set_and_check_pb_collision(q, **kwargs):
            self.robot.set_q(q)

            # Intra-robot collision detection
            for link1, link2 in self.intra_robot_link_pairs:
                if pb_utils.pairwise_link_collision(link1.body_id, link1.link_id,
                                                    link2.body_id, link2.link_id):
                    return True

            # Inter-robot collision detection
            for link1, link2 in self.inter_robot_link_pairs:
                if pb_utils.pairwise_link_collision(link1.body_id, link1.link_id,
                                                    link2.body_id, link2.link_id):
                    return True

            # Object collision detection
            for body1, body2 in self.robot_object_body_pairs:
                if pb_utils.pairwise_collision(body1, body2):
                    return True

            return False

        def for_qs(qs, per_state_f, **kwargs):
            assert qs.ndim == 2

            results = []

            for q_i in qs:
                # q_offset = 0
                # for base_pose, robot in zip(self.base_poses, self.subrobots):
                #     subrobot_q = q_i[..., q_offset:q_offset+robot.q_dim]
                #     per_robot_f(robot, q=subrobot_q, **kwargs)
                #     q_offset += robot.q_dim

                result = per_state_f(q_i, **kwargs)
                results.append(result)

            return np.array(results)



        if qs.ndim == 1:
            qs = np.expand_dims(qs, 0)
            results = for_qs(qs, set_and_check_pb_collision, **kwargs)
        elif qs.ndim == 2:
            results = for_qs(qs, set_and_check_pb_collision, **kwargs)
        elif qs.ndim == 3:

            results_l = []
            for b in qs:
                r = for_qs(b, set_and_check_pb_collision, **kwargs)
                results_l.append(r)

            results = np.stack(results_l)

        return results

    # stats

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
        colls = self.compute_collision(
            trajs_interpolated,
        )
        colls = torch.tensor(colls)
        trajs_waypoints_valid = colls.logical_not()

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

            trajs_free_inside_joint_limits_idxs = np.logical_and(
                trajs_free_tmp_position.cpu().numpy() >= self.get_q_min(),
                trajs_free_tmp_position.cpu().numpy() <= self.get_q_max()
            ).all(axis=-1).all(axis=-1)

            trajs_free_inside_joint_limits_idxs = torch.tensor(trajs_free_inside_joint_limits_idxs)
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
            **kwargs
    ):
        colls = self.compute_collision(
            trajs,
        )
        colls = torch.tensor(colls)
        fraction_free = colls.logical_not().all(axis=-1).float().mean()

        return fraction_free

    def compute_collision_intensity_trajs(
            self,
            trajs,
            **kwargs
    ):
        colls = self.compute_collision(
            trajs,
        )
        colls = torch.tensor(colls).float()
        return torch.count_nonzero(colls) / colls.nelement()

    def compute_success_free_trajs(
            self,
            trajs,
            **kwargs
    ):
        # If at least one trajectory is collision free, then we consider success
        colls = self.compute_collision(
            trajs,
        )
        any_success = torch.tensor(colls).logical_not().all(axis=-1).any()

        if any_success:
            return 1
        else:
            return 0
