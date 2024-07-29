from abc import ABC, abstractmethod

class RobotInterface(ABC):

    @property
    @abstractmethod
    def q_dim(self):
        raise NotImplementedError
