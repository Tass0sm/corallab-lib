from ..backend_manager import backend_manager


class PebbleMotionProblem:
    """A pebble motion problem from "Coordinating Pebble Motion On Graphs, The
    Diameter Of Permutation Groups, And Applications"

    """

    def __init__(
            self,
            *args,
            graph=None,
            # robot=None,
            # robot_impl=None,
            # from_impl=None,
            backend=None,
            **kwargs
    ):
        PebbleMotionProblemImpl = backend_manager.get_backend_attr(
            "PebbleMotionProblemImpl",
            backend=backend
        )

        self.problem_impl = PebbleMotionProblemImpl(
            *args,
            graph=graph,
            **kwargs
        )

    def __getattr__(self, name):
        if hasattr(self.problem_impl, name):
            return getattr(self.problem_impl, name)
        else:
            # Default behaviour
            raise AttributeError
