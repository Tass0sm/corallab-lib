import sys
from ..backend_manager import backend_manager


class MotionPlanningProblem:
    """
    A motion planning problem.
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
            **kwargs
    ):
        MotionPlanningProblemImpl = backend_manager.get_backend_attr(
            "MotionPlanningProblemImpl",
            backend=backend
        )

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

    def __getattr__(self, name):
        if hasattr(self.problem_impl, name):
            return getattr(self.problem_impl, name)
        else:
            # Default behaviour
            raise AttributeError
