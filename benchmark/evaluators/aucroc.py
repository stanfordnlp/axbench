import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
from .evaluator import Evaluator


class AUCROCEvaluator(Evaluator):
    def __init__(self, model_name):
        self.model_name = model_name
    
    def __str__(self):
        return 'AUCROCEvaluator'
    
    def compute_metrics(self, data, class_labels={"positive": 1, "negative": 0}, write_to_dir=None):
        data = data.copy()
        
        # Normalize the activation columns
        data['normalized_max'] = data[f'{self.model_name}_max_act'] / data[f'{self.model_name}_max_act'].max()
        
        # Apply class labels
        data['label'] = data['category'].map(class_labels)
        filtered_data = data.dropna(subset=['label'])
        filtered_data = filtered_data.fillna(0) # in case others are still nan, e.g., max_reax_act = 0.0
        
        # Compute ROC metrics for max_act
        fpr, tpr, thresholds = roc_curve(filtered_data['label'], filtered_data['normalized_max'])
        roc_auc = auc(fpr, tpr)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Prepare output dictionary
        metrics = {
            "roc_auc": float(roc_auc),
            "optimal_threshold": float(optimal_threshold),
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                # "thresholds": thresholds.tolist()
            }
        }
        return metrics

