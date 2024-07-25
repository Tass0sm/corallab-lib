from ..backend_manager import backend_manager


class DynamicPlanningProblem:
    def __init__(
            self,
            *args,
            env=None,
            env_impl=None,
            robot=None,
            robot_impl=None,
            backend=None,
            from_impl=None,
            **kwargs
    ):
        DynamicPlanningProblemImpl = backend_manager.get_backend_attr(
            "DynamicPlanningProblemImpl",
            backend=backend
        )

        if from_impl:
            self.problem_impl = DynamicPlanningProblemImpl.from_impl(from_impl, *args, **kwargs)
        else:
            env_impl = env_impl or env.env_impl
            robot_impl = robot_impl or robot.robot_impl

            self.problem_impl = DynamicPlanningProblemImpl(
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
