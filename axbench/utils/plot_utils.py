import os, glob, random, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.metrics import auc


def plot_aggregated_roc(metrics_list, write_to_path=None):
    # Define common FPR thresholds for interpolation
    common_fpr = np.linspace(0, 1, 100)
    
    # Initialize lists to store interpolated TPRs and AUCs for both SAE and REAX
    sae_tprs = []
    reax_tprs = []
    sae_aucs = []
    reax_aucs = []
    
    # Set up the plot
    plt.figure(figsize=(3, 3))

    all_mean_tpr = []
    all_mean_auc = []
    all_model_name = []
    
    # Process each metric in the metrics list
    tprs = {}
    aucs = {}
    for metrics in metrics_list:
        for model_name, value in metrics.items():
            fpr = value["roc_curve"]["fpr"]
            tpr = value["roc_curve"]["tpr"]
            auc = value["roc_auc"]
            
            interp_tpr = np.interp(common_fpr, fpr, tpr)
            interp_tpr[0] = 0.0  # Ensure TPR starts at 0
            if model_name not in tprs:
                tprs[model_name] = []
                aucs[model_name] = []
            tprs[model_name].append(interp_tpr)
            aucs[model_name].append(auc)

    for model_name in tprs.keys():
        # Calculate mean TPR and AUC
        mean_tpr = np.mean(tprs[model_name], axis=0)
        mean_auc = np.mean(aucs[model_name])
    
        # Plot mean ROC curve for SAE with refined style
        plt.plot(
            common_fpr, mean_tpr,
            linestyle='--',
            linewidth=2,
            label=f"{model_name} (Mean AUC = {mean_auc:.2f})"
        )
    
    # Plot diagonal line for reference with custom style
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5, label='Chance')
    
    # Aesthetic improvements for a polished look
    plt.xlabel("False Positive Rate (FPR)", fontsize=10, color='black')
    plt.ylabel("True Positive Rate (TPR)", fontsize=10, color='black')
    plt.legend(loc="lower right", fontsize=5, frameon=True, fancybox=True, framealpha=0.8, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    
    # Show the plot
    plt.savefig(write_to_path / "aggregated_roc.png", dpi=300, bbox_inches='tight')

