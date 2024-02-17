from abc import ABC, abstractmethod


class AbstractSimulator(ABC):

    @abstractmethod
    def step():
        pass
