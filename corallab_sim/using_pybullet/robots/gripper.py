class Gripper:
    """Base gripper class."""

    def __init__(self):
        self.activated = False

    def activate(self, objects):
        del objects
        return

    def release(self):
        return
