import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, precision_recall_curve
import numpy as np
from .evaluator import Evaluator


class LatentStatsEvaluator(Evaluator):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
    
    def __str__(self):
        return 'LatentStatsEvaluator'
    
    def compute_metrics(
            self, data, 
            class_labels={"positive": 1, "negative": 0, "hard negative": 0}, 
            write_to_dir=None
        ):
        data = data.copy()
        
        # Normalize the activation columns
        max_acts = data[f'{self.model_name}_max_act']
        max_act = max_acts.max()
        min_act = max_acts.min()
        data['normalized_max'] = (max_acts - min_act) / (max_act - min_act)
        data['normalized_max'] = data['normalized_max'].fillna(0)
        data['label'] = data['category'].map(class_labels)
        
        # Get threshold using only positive and negative samples
        train_data = data[data['category'].isin(['positive', 'negative'])].dropna(subset=['label'])

        # roc curve
        fpr, tpr, thresholds = roc_curve(train_data['label'], train_data['normalized_max'])
        base_auc = auc(fpr, tpr)
        j_scores = tpr - fpr
        optimal_roc_idx = np.argmax(j_scores)
        optimal_roc_threshold = thresholds[optimal_roc_idx]

        # pr curve
        precision, recall, thresholds = precision_recall_curve(train_data['label'], train_data['normalized_max'])
        base_auc = auc(recall, precision)
        # Compute F1 scores avoiding division by zero warnings
        f1_scores = np.zeros_like(precision)
        mask = (precision + recall) > 0
        f1_scores[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
        optimal_pr_idx = np.argmax(f1_scores)
        optimal_pr_threshold = thresholds[optimal_pr_idx]
        
        # Evaluate accuracy for each class
        metrics = {}
        accuracies = {}
        for category in ['positive', 'negative', 'hard negative']:
            class_data = data[data['category'] == category]
            if len(class_data) > 0:
                predictions = (class_data['normalized_max'] >= optimal_pr_threshold).astype(int)
                true_labels = class_data['label']
                accuracy = (predictions == true_labels).mean()
                accuracies[f"{category}_accuracy"] = float(accuracy)
            else:
                accuracies[f"{category}_accuracy"] = np.nan
            
        # compute precision, recall, f1 for optimal threshold
        true_labels = data['label']
        predictions = (data['normalized_max'] >= optimal_pr_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0.0)
        
        # Compute macro average accuracy over the three classes
        valid_accuracies = [acc for acc in accuracies.values()]
        metrics = {
            "positive_accuracy": float(accuracies["positive_accuracy"]),
            "negative_accuracy": float(accuracies["negative_accuracy"]),
            "hard_negative_accuracy": float(accuracies["hard negative_accuracy"]),
            "precision": float(precision) if not np.isnan(precision) else 0,
            "recall": float(recall) if not np.isnan(recall) else 0,
            "f1": float(f1) if not np.isnan(f1) else 0,
            "macro_avg_accuracy_fixed": float(np.mean(valid_accuracies)),
            "overall_accuracy": float((predictions == true_labels).mean()),
            "max_act_val": float(max_act),
            "min_act_val": float(min_act),
            "optimal_roc_threshold": float(optimal_roc_threshold),
            "optimal_pr_threshold": float(optimal_pr_threshold),
        }
        return metrics


