from abc import ABC, abstractmethod


class Evaluator(ABC):

    def fit(self, examples):
        """
        This is a placeholder in case then evaluator
        actually needs to be trained.
        """
        pass

    @abstractmethod
    def compute_metrics(self, examples):
        pass


