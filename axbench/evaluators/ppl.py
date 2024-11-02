import pandas as pd
import numpy as np
from .evaluator import Evaluator


class PerplexityEvaluator(Evaluator):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
    
    def __str__(self):
        return 'PerplexityEvaluator'
    
    def compute_metrics(self, data, write_to_dir=None):
        data = data.copy()
        metrics = {
            "perplexity": [],
            "strength": [],
            "factor": []
        }
        
        # group by factor only and compute means
        grouped = data.groupby("factor")
        for factor, group in grouped:
            perplexity = group[f"{self.model_name}_perplexity"].mean()
            metrics["perplexity"].append(perplexity)
            metrics["factor"].append(factor)
            if f"{self.model_name}_strength" in group.columns:
                strength = group[f"{self.model_name}_strength"].mean()
                metrics["strength"].append(strength)
        return metrics
