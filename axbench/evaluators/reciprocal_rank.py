import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
from .evaluator import Evaluator


class ReciprocalRankEvaluator(Evaluator):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
    
    def __str__(self):
        return 'ReciprocalRankEvaluator'
    
    def compute_metrics(
            self, data, 
            class_labels={"positive": 1, "negative": 0}, 
            write_to_dir=None
        ):
        metrics = {}
        return metrics

