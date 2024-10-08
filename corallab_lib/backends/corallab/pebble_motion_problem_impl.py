import torch
import numpy as np

from corallab_lib import MotionPlanningProblem


class PebbleMotionValidator:

    def check_motion(self, transitions):
        raise NotImplementedError


class PebbleMotionDiscreteValidator(PebbleMotionValidator):

    def check_motion(self, transitions):
        raise NotImplementedError


class PebbleMotionContinuousValidator(PebbleMotionValidator):

    def __init__(self, problem : MotionPlanningProblem):
        self.problem = problem

    def check_motion(self, transitions, **kwargs):
        q1 = torch.cat([t[0] for t in transitions])
        q2 = torch.cat([t[1] for t in transitions])

        local_motion_states = self.problem.local_motion(q1, q2, no_max_dist=True)
        collisions, info = self.problem.compute_collision_info(local_motion_states, **kwargs)
        no_collision = not collisions.any().item()

        return no_collision, info


class CorallabPebbleMotionProblem:
    def __init__(
            self,
            graph = None,
            validator = None,
            n_pebbles : int = 1,
            # robot=None,
            # impl=None,
            # tensor_args: dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        # assert isinstance(env, TorchRoboticsEnv) or env is None
        # assert isinstance(robot, TorchRoboticsRobot) or robot is None

        self.graph = graph
        self.validator = validator
        self.n_pebbles = n_pebbles

        # self.robot = robot
        # self.tensor_args = tensor_args

        # self.task_impl = PlanningTask(
        #     env=env.env_impl,
        #     robot=robot.robot_impl,
        #     tensor_args=tensor_args,
        #     **kwargs
        # )

    # def get_q_dim(self):
    #     return self.task_impl.robot.q_dim

    # def get_q_min(self):
    #     return self.task_impl.robot.q_min.cpu()

    # def get_q_max(self):
    #     return self.task_impl.robot.q_max.cpu()

    def distance_fn(self, q1, q2):
        # return np.linalg.norm(q2 - q1)
        return torch.linalg.norm(q2 - q1)

    def extend_fn(self, q1, q2, max_step=0.5, max_dist=None):
        dist = self.distance_fn(q1, q2)
        if max_dist is not None and dist > max_dist:
            q2 = q1 + (q2 - q1) * (max_dist / dist)

        alpha = np.linspace(0, 1, int(dist / max_step) + 2)
        alpha = np.expand_dims(alpha, 1)

        q1 = np.expand_dims(q1, 0)
        q2 = np.expand_dims(q2, 0)
        extension = q1 + (q2 - q1) * alpha
        return extension.squeeze()

    def check_motion(self, transitions, **kwargs):
        return self.validator.check_motion(transitions, **kwargs)

    # def random_q(self, **kwargs):
    #     return self.task_impl.sample_q(without_collision=False, **kwargs)

    # def random_coll_free_q(self, *args, **kwargs):
    #     return self.task_impl.random_coll_free_q(*args, **kwargs)

    # def compute_collision_info(self, q, **kwargs):
    #     q = to_torch(q, **self.tensor_args)
    #     return self.task_impl.compute_collision_info(q, **kwargs)

    # def compute_collision(self, q, **kwargs):
    #     q = to_torch(q, **self.tensor_args)
    #     return self.task_impl.compute_collision(q, **kwargs)

    # def get_trajs_collision_and_free(self, trajs, **kwargs):
    #     return self.task_impl.get_trajs_collision_and_free(trajs, **kwargs)

    # def compute_fraction_free_trajs(self, trajs, **kwargs):
    #     return self.task_impl.compute_fraction_free_trajs(trajs, **kwargs)

    # def compute_collision_intensity_trajs(self, trajs, **kwargs):
    #     return self.task_impl.compute_collision_intensity_trajs(trajs, **kwargs)

    # def compute_success_free_trajs(self, trajs, **kwargs):
    #     return self.task_impl.compute_success_free_trajs(trajs, **kwargs)
