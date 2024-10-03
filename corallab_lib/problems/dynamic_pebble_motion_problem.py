from ..backend_manager import backend_manager


class DynamicPebbleMotionProblem:
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
        DynamicPebbleMotionProblemImpl = backend_manager.get_backend_attr(
            "DynamicPebbleMotionProblemImpl",
            backend=backend
        )

        if from_impl:
            self.problem_impl = DynamicPebbleMotionProblemImpl.from_impl(
                from_impl,
                *args,
                env=env.env_impl,
                robot=robot.robot_impl,
                **kwargs
            )
        else:
            env_impl = env_impl or env.entity_impl
            robot_impl = robot_impl or robot.entity_impl

            self.problem_impl = DynamicPebbleMotionProblemImpl(
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
