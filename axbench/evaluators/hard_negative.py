import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
from .evaluator import Evaluator


class HardNegativeEvaluator(Evaluator):
    """
    Evaluator that computes the accuracy for positive, negative, and hard negative classes using the optimal
    threshold from positive and negative samples, and then calculates the macro average accuracy.
    """
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
    
    def __str__(self):
        return 'HardNegativeEvaluator'
    
    def compute_metrics(
            self, data, 
            class_labels={"positive": 1, "negative": 0, "hard negative": 0}, 
            write_to_dir=None
        ):
        data = data.copy()
        
        # Normalize the activation columns
        max_acts = data[f'{self.model_name}_max_act']
        data['normalized_max'] = (max_acts - max_acts.min()) / (max_acts.max() - max_acts.min())
        data['normalized_max'] = data['normalized_max'].fillna(0)
        data['label'] = data['category'].map(class_labels)
        
        # Get threshold using only positive and negative samples
        train_data = data[data['category'].isin(['positive', 'negative'])].dropna(subset=['label'])
        fpr, tpr, thresholds = roc_curve(train_data['label'], train_data['normalized_max'])
        base_auc = auc(fpr, tpr)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # If threshold is positive or negative infinity, return 0 for all metrics
        if np.isinf(float(optimal_threshold)):
            return {}
        
        # Evaluate accuracy for each class
        metrics = {}
        accuracies = {}
        for category in ['positive', 'hard negative']:
            class_data = data[data['category'] == category]
            if len(class_data) > 0:
                predictions = (class_data['normalized_max'] >= optimal_threshold).astype(int)
                true_labels = class_data['label']
                accuracy = (predictions == true_labels).mean()
                accuracies[f"{category}_accuracy"] = float(accuracy)
            else:
                return {}
        
        # Compute macro average accuracy over the three classes
        valid_accuracies = [acc for acc in accuracies.values()]
        metrics["macro_avg_accuracy"] = float(np.mean(valid_accuracies))
        return metrics


