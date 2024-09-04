import torch
import numpy as np
import corallab_lib.backends.pybullet.ompl.utils as pb_utils

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch, to_numpy

from itertools import product

import scipy.interpolate

from corallab_lib import Env, Robot
from .env_impl import PybulletEnv
from .robot_impl import PybulletRobot
from .motion_planning_problem_impl import PybulletMotionPlanningProblem


class DynamicObstacle:

    def __init__(
            self,
            entity,
            traj,
            timesteps,
            tensor_args : dict = DEFAULT_TENSOR_ARGS
    ):
        if isinstance(traj, np.ndarray):
            traj = to_torch(traj, **tensor_args)

        if isinstance(timesteps, np.ndarray):
            timesteps = to_torch(timesteps, **tensor_args)

        if traj.ndim == 2:
            traj = traj.unsqueeze(0)

        assert timesteps.shape[0] == traj.shape[1]

        self.entity = entity
        self.traj = traj.to(**tensor_args)
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

    def set_q(
            self,
            time : int,
            **kwargs
    ):
        q = self.dynamic_state(time)
        self.entity.set_q(q, **kwargs)

    # def get_margins(self):
    #     margins = to_torch(self.entity.link_margins_for_object_collision_checking, **self.tensor_args)
    #     return margins


class PybulletDynamicPlanningProblem(PybulletMotionPlanningProblem):
    def __init__(
            self,
            t_max : float = 64.0,
            v_max : float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.t_max = t_max
        self.v_max = v_max

        self.dynamic_obstacles = []
        self._setup_dynamic_collision_detection()

    def add_dynamic_obstacle(self, robot, traj, time):
        obstacle = DynamicObstacle(robot, traj, time, tensor_args=self.tensor_args)
        self.dynamic_obstacles.append(obstacle)
        self._setup_dynamic_collision_detection()

    def clear_dynamic_obstacles(self):
        self.dynamic_obstacles = []
        self._setup_dynamic_collision_detection()

    def _setup_dynamic_collision_detection(self):
        pybullet_robots = [self.robot.robot_impl]

        self.robot_dynamic_object_body_pairs = []

        moving_bodies = []
        all_moving_links = []
        self.inter_robot_link_pairs = []
        for pybullet_bot in pybullet_robots:
            bot_moving_links = pb_utils.get_moving_links(
                pybullet_bot.id,
                pybullet_bot.arm_controllable_joints
            )
            full_bot_moving_links = [pb_utils.Link(pybullet_bot.id, x) for x in bot_moving_links]

            for dynamic_obstacle in self.dynamic_obstacles:
                dynamic_obstacle_links = pb_utils.get_moving_links(
                    dynamic_obstacle.id,
                    dynamic_obstacle.arm_controllable_joints
                )
                full_dynamic_obstacle_links = [pb_utils.Link(dynamic_obstacle.id, x) for x in dynamic_obstacle_links]

                self.robot_dynamic_object_body_pairs.extend(
                    product(full_bot_moving_links, full_dynamic_obstacle_links)
                )

    def static_check_collision(self, qs, **kwargs):
        return super().check_collision(qs, **kwargs)

    def _position_dynamic_obstacles(self, time):
        for obstacle in self.dynamic_obstacles:
            obstacle.set_q(time)

    def check_collision(self, time, qs, **kwargs):
        if isinstance(qs, torch.Tensor):
            qs = qs.cpu().numpy()



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

            # Dynamic object collision detection
            for body1, body2 in self.robot_dynamic_object_body_pairs:
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
