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
    
    # Process each metric in the metrics list
    for metrics in metrics_list:
        # Extract and interpolate SAE metrics
        sae_fpr = metrics["sae"]["roc_curve"]["fpr"]
        sae_tpr = metrics["sae"]["roc_curve"]["tpr"]
        sae_auc = metrics["sae"]["roc_auc"]
        
        interp_sae_tpr = np.interp(common_fpr, sae_fpr, sae_tpr)
        interp_sae_tpr[0] = 0.0  # Ensure TPR starts at 0
        sae_tprs.append(interp_sae_tpr)
        sae_aucs.append(sae_auc)
        
        # Extract and interpolate REAX metrics
        reax_fpr = metrics["reax"]["roc_curve"]["fpr"]
        reax_tpr = metrics["reax"]["roc_curve"]["tpr"]
        reax_auc = metrics["reax"]["roc_auc"]
        
        interp_reax_tpr = np.interp(common_fpr, reax_fpr, reax_tpr)
        interp_reax_tpr[0] = 0.0  # Ensure TPR starts at 0
        reax_tprs.append(interp_reax_tpr)
        reax_aucs.append(reax_auc)
    
    # Calculate mean TPR and AUC for both SAE and REAX
    mean_sae_tpr = np.mean(sae_tprs, axis=0)
    mean_sae_auc = np.mean(sae_aucs)
    mean_reax_tpr = np.mean(reax_tprs, axis=0)
    mean_reax_auc = np.mean(reax_aucs)
    
    # Plot mean ROC curve for SAE with refined style
    plt.plot(
        common_fpr, mean_sae_tpr,
        color='royalblue',
        linestyle='--',
        linewidth=2,
        label=f"SAE (Mean AUC = {mean_sae_auc:.2f})"
    )
    
    # Plot mean ROC curve for REAX with refined style
    plt.plot(
        common_fpr, mean_reax_tpr,
        color='darkorange',
        linestyle='-',
        linewidth=2,
        label=f"ReAX (Mean AUC = {mean_reax_auc:.2f})"
    )
    
    # Plot diagonal line for reference with custom style
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5, label='Chance')
    
    # Aesthetic improvements for a polished look
    plt.xlabel("False Positive Rate (FPR)", fontsize=10, color='black')
    plt.ylabel("True Positive Rate (TPR)", fontsize=10, color='black')
    plt.legend(loc="lower right", fontsize=8, frameon=True, fancybox=True, framealpha=0.8, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    
    # Show the plot
    plt.savefig(write_to_path / "aggregated_roc.png", dpi=300, bbox_inches='tight')
    
    # Return the average AUC values for SAE and REAX for verification
    return {"mean_sae_auc": mean_sae_auc, "mean_reax_auc": mean_reax_auc}

