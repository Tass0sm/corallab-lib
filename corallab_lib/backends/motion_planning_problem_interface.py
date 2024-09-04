from typing import Optional
from jaxtyping import Array, Float, Bool
from abc import ABC, abstractmethod

class MotionPlanningProblemInterface(ABC):

    @abstractmethod
    def check_collision(
            self,
            q: Float[Array, "b h d"],
            margin : Optional[float] = None,
            **kwargs
    ) -> Bool[Array, "b h"]:
        raise NotImplementedError
