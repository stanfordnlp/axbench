import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
from .evaluator import Evaluator


class AUCROCEvaluator(Evaluator):
    def __init__(self):
        pass
    
    def __str__(self):
        return 'AUCROCEvaluator'
    
    def compute_metrics(self, data, class_labels={"positive": 1, "negative": 0}, write_to_dir=None):
        data = data.copy()
        
        # Normalize the activation columns
        data['normalized_max_sae'] = data['max_sae_act'] / data['max_sae_act'].max()
        data['normalized_max_reax'] = data['max_reax_act'] / data['max_reax_act'].max()
        
        # Apply class labels
        data['label'] = data['category'].map(class_labels)
        filtered_data = data.dropna(subset=['label'])
        
        # Compute ROC metrics for max_sae_act
        fpr_sae, tpr_sae, thresholds_sae = roc_curve(filtered_data['label'], filtered_data['normalized_max_sae'])
        roc_auc_sae = auc(fpr_sae, tpr_sae)
        j_scores_sae = tpr_sae - fpr_sae
        optimal_idx_sae = np.argmax(j_scores_sae)
        optimal_threshold_sae = thresholds_sae[optimal_idx_sae]
        
        # Compute ROC metrics for max_reax_act
        fpr_reax, tpr_reax, thresholds_reax = roc_curve(filtered_data['label'], filtered_data['normalized_max_reax'])
        roc_auc_reax = auc(fpr_reax, tpr_reax)
        j_scores_reax = tpr_reax - fpr_reax
        optimal_idx_reax = np.argmax(j_scores_reax)
        optimal_threshold_reax = thresholds_reax[optimal_idx_reax]
        
        # Prepare output dictionary
        metrics = {
            "sae": {
                "roc_auc": float(roc_auc_sae),
                "optimal_threshold": float(optimal_threshold_sae),
                "roc_curve": {
                    "fpr": fpr_sae.tolist(),
                    "tpr": tpr_sae.tolist(),
                    # "thresholds": thresholds_sae.tolist()
                }
            },
            "reax": {
                "roc_auc": float(roc_auc_reax),
                "optimal_threshold": float(optimal_threshold_reax),
                "roc_curve": {
                    "fpr": fpr_reax.tolist(),
                    "tpr": tpr_reax.tolist(),
                    # "thresholds": thresholds_reax.tolist()
                }
            }
        }
        
        return metrics
