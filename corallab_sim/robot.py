from .backend_manager import backend_manager

class Robot:
    def __init__(self, id: str):
        RobotImpl = backend_manager.get_backend_attr("RobotImpl")
        self.robot_impl = RobotImpl(id)

    def __getattr__(self, name):
        if hasattr(self.robot_impl, name):
            return getattr(self.robot_impl, name)
        else:
            # Default behaviour
            raise AttributeError
