import os, glob, random, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.metrics import auc
from matplotlib.colors import hsv_to_rgb

# Predefined color and marker sequences for consistency
COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf'   # cyan
]

MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']


def plot_aggregated_roc(metrics_list, write_to_path=None):
    # Define common FPR thresholds for interpolation
    common_fpr = np.linspace(0, 1, 100)

    # Set up the plot
    plt.figure(figsize=(3, 3))
    
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

    # Sort model names for consistent ordering
    sorted_models = sorted(tprs.keys())
    
    # Plot each model's ROC curve with consistent styling
    for idx, model_name in enumerate(sorted_models):
        mean_tpr = np.mean(tprs[model_name], axis=0)
        mean_auc = np.mean(aucs[model_name])
        
        plt.plot(
            common_fpr, mean_tpr,
            color=COLORS[idx % len(COLORS)],
            linestyle='--',
            linewidth=2,
            label=f"{model_name} (Mean AUC = {mean_auc:.2f})"
        )
    
    # Rest of the plotting code remains the same
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5, label='Chance')
    
    plt.xlabel("False Positive Rate (FPR)", fontsize=10, color='black')
    plt.ylabel("True Positive Rate (TPR)", fontsize=10, color='black')
    plt.legend(loc="lower right", fontsize=5, frameon=True, fancybox=True, framealpha=0.8, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    
    plt.savefig(write_to_path / "aggregated_roc.png", dpi=300, bbox_inches='tight')


def plot_metric(jsonl_data, evaluator_name, metric_name, y_label, use_log_scale=False, write_to_path=None):
    # Get unique methods and sort them
    methods = set()
    for entry in jsonl_data:
        methods.update(entry['results'][evaluator_name].keys())
    methods = sorted(list(methods))
    
    # Aggregate data across concepts
    aggregated = {method: {
        metric_name: [],
        'factor': []
    } for method in methods}
    
    # Collect data from all concepts
    for entry in jsonl_data:
        results = entry['results'][evaluator_name]
        for method in methods:
            if method in results:
                aggregated[method][metric_name].append(results[method][metric_name])
                aggregated[method]['factor'].append(results[method]['factor'])
    
    # Average across concepts
    for method in methods:
        aggregated[method][metric_name] = np.mean(aggregated[method][metric_name], axis=0)
        aggregated[method]['factor'] = aggregated[method]['factor'][0]

    # Create the plot
    plt.figure(figsize=(6, 3))
    
    # Plot the metric with consistent colors and markers
    for idx, method in enumerate(methods):
        plt.plot(aggregated[method]['factor'], 
                aggregated[method][metric_name],
                color=COLORS[idx % len(COLORS)],
                marker=MARKERS[idx % len(MARKERS)],
                linestyle='--',
                linewidth=2,
                markersize=6,
                markeredgecolor='black',
                markeredgewidth=1,
                label=f'{method}')
    
    # Rest of the customization remains the same
    plt.xlabel('Factor', fontsize=10, color='black')
    plt.ylabel(y_label, fontsize=10, color='black')
    
    if use_log_scale:
        plt.yscale('log')
    
    first_method = methods[0]
    plt.xticks(aggregated[first_method]['factor'], fontsize=6, color='black')
    plt.yticks(fontsize=8, color='black')
    
    plt.legend(loc="lower right", 
              fontsize=5, 
              frameon=True, 
              fancybox=True, 
              framealpha=0.8, 
              shadow=True)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if write_to_path:
        plt.savefig(write_to_path / f"{metric_name}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()