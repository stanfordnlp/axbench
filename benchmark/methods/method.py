from abc import ABC, abstractmethod


class Method(ABC):

    @abstractmethod
    def train(self, examples):
        pass

    @abstractmethod
    def predict(self, examples):
        pass
