from .evaluator import Evaluator


class EloEvaluator(Evaluator):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.lm_model = kwargs.get("lm_model", None)

    def __str__(self):
        return 'EloEvaluator'

    def compute_metrics(self, examples):
        pass


