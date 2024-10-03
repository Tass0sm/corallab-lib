from typing import Optional
from abc import ABC, abstractmethod

class MotionPlanningProblemInterface(ABC):

    @abstractmethod
    def check_collision(
            self,
            q,
            margin : Optional[float] = None,
            **kwargs
    ):
        raise NotImplementedError
