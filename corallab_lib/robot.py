from .backend_manager import backend_manager

class Robot:
    def __init__(
            self,
            *args,
            backend=None,
            from_impl=None,
            **kwargs
    ):
        RobotImpl = backend_manager.get_backend_attr(
            "RobotImpl",
            backend=backend
        )

        if from_impl:
            self.robot_impl = RobotImpl.from_impl(from_impl, *args, **kwargs)
        else:
            self.robot_impl = RobotImpl(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.robot_impl, name):
            return getattr(self.robot_impl, name)
        else:
            # Default behaviour
            raise AttributeError

    def __eq__(self, other):
        return self.robot_impl.robot_id == other.robot_impl.robot_id
