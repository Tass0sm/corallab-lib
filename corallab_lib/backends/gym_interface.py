from abc import ABC, abstractmethod

class GymInterface(ABC):

    @property
    @abstractmethod
    def state_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self):
        raise NotImplementedError
