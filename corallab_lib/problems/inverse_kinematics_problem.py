import sys
from ..backend_manager import backend_manager

# import torch
# from jaxtyping import Array, Float, Bool


class InverseKinematicsProblem:
    """An inverse kinematics problem.

    """

    def __init__(
            self,
            # env=None,
            # env_impl=None,
            robot=None,
            # robot_impl=None,
            goal_poses=None,
            retract_config=None,
            # from_impl=None,
            # backend=None,
            **kwargs
    ):
        self.robot = robot
        self.goal_poses = goal_poses

        self.goal_poses = goal_poses
        self.retract_config = retract_config

        # MotionPlanningProblemImpl = backend_manager.get_backend_attr(
        #     "MotionPlanningProblemImpl",
        #     backend=backend
        # )

        # if from_impl:
        #     self.problem_impl = MotionPlanningProblemImpl.from_impl(
        #         from_impl,
        #         *args,
        #         env=env.env_impl,
        #         robot=robot.robot_impl,
        #         **kwargs
        #     )
        # else:
        #     env_impl = env_impl or env.env_impl
        #     robot_impl = robot_impl or robot.robot_impl

        #     self.problem_impl = MotionPlanningProblemImpl(
        #         *args,
        #         env=env_impl,
        #         robot=robot_impl,
        #         **kwargs
        #     )
        pass

    # def __getattr__(self, name):
    #     if hasattr(self.problem_impl, name):
    #         return getattr(self.problem_impl, name)
    #     else:
    #         # Default behaviour
    #         raise AttributeError

    # def check_solutions(self, q: Float[Array, "b h d"], **kwargs) -> Bool[Array, "b"]:
    #     return self.check_collision(q, **kwargs).logical_not().all(axis=-1)

    # def check_any_valid(self, q: Float[Array, "b h d"], **kwargs) -> Bool[Array, "b"]:
    #     return self.check_solutions(q, **kwargs).any()

    # def compute_fraction_valid(self, q: Float[Array, "b h d"], **kwargs) -> Bool[Array, "b"]:
    #     return self.check_solutions(q, **kwargs).float().mean()

    # def get_valid_and_invalid(
    #         self,
    #         q: Float[Array, "b h d"],
    #         **kwargs
    # ) -> (Bool[Array, "b"], Bool[Array, "b"]):
    #     validity = self.check_solutions(q, **kwargs)

    #     valid_idxs = torch.argwhere(validity)
    #     invalid_idxs = torch.argwhere(validity.logical_not())

    #     trajs_valid = q[valid_idxs.squeeze(), ...]
    #     if trajs_valid.ndim == 2:
    #         trajs_valid = trajs_valid.unsqueeze(0)

    #     trajs_invalid = q[invalid_idxs.squeeze(), ...]
    #     if trajs_invalid.ndim == 2:
    #         trajs_invalid = trajs_invalid.unsqueeze(0)

    #     if trajs_valid.nelement() == 0:
    #         trajs_valid = None

    #     if trajs_invalid.nelement() == 0:
    #         trajs_invalid = None

    #     return trajs_valid, trajs_invalid
