import sys
from ..backend_manager import backend_manager

import torch
import numpy as np
from jaxtyping import Array, Float, Bool

from torch_robotics.torch_utils.torch_utils import to_torch
from torch_robotics.trajectory.utils import interpolate_traj_via_points


class MotionPlanningProblem:
    """A motion planning problem.

    """

    def __init__(
            self,
            *args,
            env=None,
            env_impl=None,
            robot=None,
            robot_impl=None,
            from_impl=None,
            backend=None,
            threshold_start_goal_pos=1.0,
            start_state_pos=None,
            goal_state_pos=None,
            **kwargs
    ):
        MotionPlanningProblemImpl = backend_manager.get_backend_attr(
            "MotionPlanningProblemImpl",
            backend=backend
        )

        # TODO: check
        if backend is not None:
            self.backend = backend
        else:
            self.backend = backend_manager.backend

        if from_impl:
            self.problem_impl = MotionPlanningProblemImpl.from_impl(
                from_impl,
                *args,
                env=env.env_impl,
                robot=robot.robot_impl,
                **kwargs
            )
        else:
            env_impl = env_impl or env.env_impl
            robot_impl = robot_impl or robot.robot_impl

            self.problem_impl = MotionPlanningProblemImpl(
                *args,
                env=env_impl,
                robot=robot_impl,
                **kwargs
            )

        # Generate random start and goal
        if start_state_pos is None or goal_state_pos is None:
            random_start_pos, random_goal_pos = self._generate_random_start_and_goal(threshold_start_goal_pos)

        self.start_state_pos = start_state_pos if start_state_pos is not None else random_start_pos
        self.goal_state_pos = goal_state_pos if goal_state_pos is not None else random_goal_pos

    def __getattr__(self, name):
        if hasattr(self.problem_impl, name):
            return getattr(self.problem_impl, name)
        else:
            # Default behaviour
            raise AttributeError

    def _generate_random_start_and_goal(
            self,
            threshold_start_goal_pos,
            n_tries=1000,
    ):
        """
        Generate random start and goal states.
        """
        print("Generating random start and goal...")

        start_state_pos, goal_state_pos = None, None
        for _ in range(n_tries):
            q_free, _ = self.random_coll_free_q(n_samples=2)

            start_state_pos = q_free[0]
            goal_state_pos = q_free[1]

            if isinstance(start_state_pos, np.ndarray):
                start_state_pos = to_torch(start_state_pos, **self.problem_impl.tensor_args)

            if isinstance(goal_state_pos, np.ndarray):
                goal_state_pos = to_torch(goal_state_pos, **self.problem_impl.tensor_args)

            if torch.linalg.norm(start_state_pos - goal_state_pos) > threshold_start_goal_pos:
                break

        if start_state_pos is None or goal_state_pos is None:
            raise ValueError(f"No collision free configuration was found\n"
                             f"start_state_pos: {start_state_pos}\n"
                             f"goal_state_pos:  {goal_state_pos}\n")

        return start_state_pos, goal_state_pos

    def check_solutions(
            self,
            q: Float[Array, "b h d"],
            num_interpolation : int = 5,
            **kwargs
    ) -> Bool[Array, "b"]:
        q_interp = interpolate_traj_via_points(q, num_interpolation=num_interpolation)
        colls = self.check_collision(q_interp, margin=0.0, **kwargs)
        colls = to_torch(colls)
        return colls.logical_not().all(axis=-1)

    def check_any_valid(self, q: Float[Array, "b h d"], **kwargs) -> Bool[Array, "b"]:
        return self.check_solutions(q, **kwargs).any()

    def compute_fraction_valid(self, q: Float[Array, "b h d"], **kwargs) -> Bool[Array, "b"]:
        return self.check_solutions(q, **kwargs).float().mean()

    def get_valid_and_invalid(
            self,
            q : Float[Array, "b h d"],
            return_indices : bool = False,
            **kwargs
    ) -> (Bool[Array, "b"], Bool[Array, "b"]):
        validity = self.check_solutions(q, **kwargs)

        valid_idxs = torch.argwhere(validity)
        invalid_idxs = torch.argwhere(validity.logical_not())

        trajs_valid = q[valid_idxs.squeeze(), ...]
        if trajs_valid.ndim == 2:
            trajs_valid = trajs_valid.unsqueeze(0)

        trajs_invalid = q[invalid_idxs.squeeze(), ...]
        if trajs_invalid.ndim == 2:
            trajs_invalid = trajs_invalid.unsqueeze(0)

        if trajs_valid.nelement() == 0:
            trajs_valid = None

        if trajs_invalid.nelement() == 0:
            trajs_invalid = None

        if return_indices:
            return trajs_valid, valid_idxs, trajs_invalid, invalid_idxs
        else:
            return trajs_valid, trajs_invalid
