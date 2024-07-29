import torch
from jaxtyping import Array, Float, Bool

from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch, to_numpy
from .env_impl import TorchRoboticsEnv
from .robot_impl import TorchRoboticsRobot

from ..motion_planning_problem_interface import MotionPlanningProblemInterface


class TorchRoboticsMotionPlanningProblem(MotionPlanningProblemInterface):
    def __init__(
            self,
            env=None,
            robot=None,
            impl=None,
            local_collision_step : float = 0.03,
            local_collision_max_dist : float = 0.1,
            obstacle_cutoff_margin : float = 0.005,
            tensor_args: dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        assert isinstance(env, TorchRoboticsEnv) or env is None
        assert isinstance(robot, TorchRoboticsRobot) or robot is None

        self.env = env
        self.robot = robot
        self.tensor_args = tensor_args

        self.local_collision_step = local_collision_step
        self.local_collision_max_dist = local_collision_max_dist

        if impl:
            self.task_impl = impl
        else:
            self.task_impl = PlanningTask(
                env=env.env_impl,
                robot=robot.robot_impl,
                obstacle_cutoff_margin=obstacle_cutoff_margin,
                tensor_args=tensor_args,
                **kwargs
            )

    @classmethod
    def from_impl(cls, impl, **kwargs):
        # id is None
        return cls(None, impl=impl, **kwargs)

    def get_q_dim(self):
        return self.task_impl.robot.q_dim

    def get_q_min(self):
        return self.task_impl.robot.q_min.cpu()

    def get_q_max(self):
        return self.task_impl.robot.q_max.cpu()

    def distance_q(self, q1, q2):
        return self.task_impl.distance_q(q1, q2)

    def random_q(self, **kwargs):
        return self.task_impl.sample_q(without_collision=False, **kwargs)

    def random_coll_free_q(self, *args, **kwargs):
        return self.task_impl.random_coll_free_q(*args, **kwargs)

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

    def compute_collision_info(self, q, **kwargs):
        q = to_torch(q, **self.tensor_args)
        return self.task_impl.compute_collision_info(q, **kwargs)

    def check_collision(self, q: Float[Array, "b h d"], **kwargs) -> Bool[Array, "b h"]:
        q = to_torch(q, **self.tensor_args)
        return self.task_impl.compute_collision(q, **kwargs)

    def get_trajs_collision_and_free(self, trajs, **kwargs):
        return self.task_impl.get_trajs_collision_and_free(trajs, **kwargs)

    def compute_fraction_free_trajs(self, trajs, **kwargs):
        return self.task_impl.compute_fraction_free_trajs(trajs, **kwargs)

    def compute_collision_intensity_trajs(self, trajs, **kwargs):
        return self.task_impl.compute_collision_intensity_trajs(trajs, **kwargs)

    def compute_success_free_trajs(self, trajs, **kwargs):
        return self.task_impl.compute_success_free_trajs(trajs, **kwargs)
