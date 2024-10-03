from .entity import Entity


class Robot(Entity):
    def __init__(
            self,
            *args,
            backend=None,
            from_impl=None,
            **kwargs
    ):
        super().__init__("RobotImpl", *args, backend=backend, from_impl=from_impl, **kwargs)

    def __eq__(self, other):
        return self.robot_impl.robot_id == other.robot_impl.robot_id
