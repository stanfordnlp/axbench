import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
from .evaluator import Evaluator


class HardNegativeEvaluator(Evaluator):
    """
    WARNING: it turns out pure accuracy on hard negatives is not a good metric for evaluating abstractions.
    The abstraction could be very insensitive to any concept, but still have high accuracy on hard negatives.
    This might also affect many similar metrics in other papers on ambiguity, fuzziness, etc. Be careful out there :)
    """
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
    
    def __str__(self):
        return 'HardNegativeEvaluator'
    
    def compute_metrics(
            self, data, 
            class_labels={"positive": 1, "negative": 0, "hard negative seen": 0, "hard negative unseen": 0}, 
            write_to_dir=None
        ):
        data = data.copy()
        
        # Normalize the activation columns
        max_acts = data[f'{self.model_name}_max_act']
        data['normalized_max'] = (max_acts - max_acts.min()) / (max_acts.max() - max_acts.min())
        data['normalized_max'] = data['normalized_max'].fillna(0)
        data['label'] = data['category'].map(class_labels)
        
        # First, get threshold using only positive and negative samples
        train_data = data[data['category'].isin(['positive', 'negative'])].dropna(subset=['label'])
        fpr, tpr, thresholds = roc_curve(train_data['label'], train_data['normalized_max'])
        base_auc = auc(fpr, tpr)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # If threshold is positive or negative infinity, return 0 for all metrics
        if np.isinf(float(optimal_threshold)):
            return {
                "hard_negative_seen_accuracy": 0.0,
                "hard_negative_unseen_accuracy": 0.0,
                "base_auc": base_auc
            }
        
        # Then evaluate hard negatives
        hard_neg_seen = data[data['category'] == "hard negative seen"]
        hard_neg_unseen = data[data['category'] == "hard negative unseen"]
        
        metrics = {"base_auc": float(base_auc)}
        
        # Compute accuracy (percentage correctly classified as negative)
        if len(hard_neg_seen) > 0:
            predictions = (hard_neg_seen['normalized_max'] >= optimal_threshold).astype(int)
            metrics["hard_negative_seen_accuracy"] = float((predictions == 0).mean()) * base_auc
        
        if len(hard_neg_unseen) > 0:
            predictions = (hard_neg_unseen['normalized_max'] >= optimal_threshold).astype(int)
            metrics["hard_negative_unseen_accuracy"] = float((predictions == 0).mean()) * base_auc
        
        return metrics

