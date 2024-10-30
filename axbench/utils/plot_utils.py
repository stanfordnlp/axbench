import os, glob, random, json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.metrics import auc
from matplotlib.colors import hsv_to_rgb


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


def plot_perplexity(jsonl_data, write_to_path=None):
    """
    Plot perplexity vs factor for different methods.
    Data is aggregated over concepts.
    
    Args:
        jsonl_data: List of dictionaries containing evaluation results
        write_to_path: Optional path to save the plot
    """    
    # Get unique methods from the data
    methods = set()
    for entry in jsonl_data:
        methods.update(entry['results']['PerplexityEvaluator'].keys())
    methods = sorted(list(methods))
    
    # Generate evenly spaced colors based on number of methods
    colors = [hsv_to_rgb((i / len(methods), 0.8, 0.8)) for i in range(len(methods))]
    
    # Aggregate data across concepts
    aggregated = {method: {
        'perplexity': [],
        'factor': []
    } for method in methods}
    
    # Collect data from all concepts
    for entry in jsonl_data:
        results = entry['results']['PerplexityEvaluator']
        for method in methods:
            if method in results:
                aggregated[method]['perplexity'].append(results[method]['perplexity'])
                aggregated[method]['factor'].append(results[method]['factor'])
    
    # Average across concepts
    for method in methods:
        aggregated[method]['perplexity'] = np.mean(aggregated[method]['perplexity'], axis=0)
        aggregated[method]['factor'] = aggregated[method]['factor'][0]  # Take first since all same
    
    # Create the plot
    plt.figure(figsize=(6, 3))
    
    # Define marker styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']  # Add more if needed
    
    # Plot perplexity
    for method, color, marker in zip(methods, colors, markers[:len(methods)]):
        plt.plot(aggregated[method]['factor'], 
                aggregated[method]['perplexity'],
                linestyle='--',
                linewidth=2,
                marker=marker,
                markersize=6,
                markerfacecolor=color,
                markeredgecolor='black',
                markeredgewidth=1,
                label=f'{method}',
                color=color)
    
    # Customize the plot
    plt.xlabel('Factor', fontsize=10, color='black')
    plt.ylabel('Perplexity', fontsize=10, color='black')
    
    # Set y-axis to log scale for perplexity
    plt.yscale('log')
    
    # Add finer x-axis ticks
    min_factor = min(min(aggregated[method]['factor']) for method in methods)
    max_factor = max(max(aggregated[method]['factor']) for method in methods)
    plt.xticks(np.arange(min_factor, max_factor + 0.5, 0.5), fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    
    # Add legend
    plt.legend(loc="lower right", 
              fontsize=5, 
              frameon=True, 
              fancybox=True, 
              framealpha=0.8, 
              shadow=True)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if write_to_path:
        plt.savefig(write_to_path / "perplexity.png", 
                   dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_strength(jsonl_data, write_to_path=None):  
    # Get unique methods from the data
    methods = set()
    for entry in jsonl_data:
        methods.update(entry['results']['PerplexityEvaluator'].keys())
    methods = sorted(list(methods))
    
    # Generate evenly spaced colors based on number of methods
    colors = [hsv_to_rgb((i / len(methods), 0.8, 0.8)) for i in range(len(methods))]
    
    # Aggregate data across concepts
    aggregated = {method: {
        'strength': [],
        'factor': []
    } for method in methods}
    
    # Collect data from all concepts
    for entry in jsonl_data:
        results = entry['results']['PerplexityEvaluator']
        for method in methods:
            if method in results:
                aggregated[method]['strength'].append(results[method]['strength'])
                aggregated[method]['factor'].append(results[method]['factor'])

    # Average across concepts
    for method in methods:
        aggregated[method]['strength'] = np.mean(aggregated[method]['strength'], axis=0)
        aggregated[method]['factor'] = aggregated[method]['factor'][0]  # Take first since all same

    # Create the plot
    plt.figure(figsize=(6, 3))
    
    # Define marker styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']  # Add more if needed
    
    # Plot perplexity
    for method, color, marker in zip(methods, colors, markers[:len(methods)]):
        plt.plot(aggregated[method]['factor'], 
                aggregated[method]['strength'],
                linestyle='--',
                linewidth=2,
                marker=marker,
                markersize=6,
                markerfacecolor=color,
                markeredgecolor='black',
                markeredgewidth=1,
                label=f'{method}',
                color=color)
    
    # Customize the plot
    plt.xlabel('Factor', fontsize=10, color='black')
    plt.ylabel('Strength', fontsize=10, color='black')
    
    # Add finer x-axis ticks
    min_factor = min(min(aggregated[method]['factor']) for method in methods)
    max_factor = max(max(aggregated[method]['factor']) for method in methods)
    plt.xticks(np.arange(min_factor, max_factor + 0.5, 0.5), fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    
    # Add legend
    plt.legend(loc="lower right", 
              fontsize=5, 
              frameon=True, 
              fancybox=True, 
              framealpha=0.8, 
              shadow=True)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if write_to_path:
        plt.savefig(write_to_path / "strength.png", 
                   dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_lm_judge_rating(jsonl_data, write_to_path=None):
    # Get unique methods from the data
    methods = set()
    for entry in jsonl_data:
        methods.update(entry['results']['LMJudgeEvaluator'].keys())
    methods = sorted(list(methods))
    
    # Generate evenly spaced colors based on number of methods
    colors = [hsv_to_rgb((i / len(methods), 0.8, 0.8)) for i in range(len(methods))]
    
    # Aggregate data across concepts
    aggregated = {method: {
        'lm_judge_rating': [],
        'factor': []
    } for method in methods}
    
    # Collect data from all concepts
    for entry in jsonl_data:
        results = entry['results']['LMJudgeEvaluator']
        for method in methods:
            if method in results:
                aggregated[method]['lm_judge_rating'].append(results[method]['lm_judge_rating'])
                aggregated[method]['factor'].append(results[method]['factor'])
    
    # Average across concepts
    for method in methods:
        aggregated[method]['lm_judge_rating'] = np.mean(aggregated[method]['lm_judge_rating'], axis=0)
        aggregated[method]['factor'] = aggregated[method]['factor'][0]  # Take first since all same
    
    # Create the plot
    plt.figure(figsize=(6, 3))
    
    # Define marker styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']  # Add more if needed
    
    # Plot lm_judge_rating
    for method, color, marker in zip(methods, colors, markers[:len(methods)]):
        plt.plot(aggregated[method]['factor'], 
                aggregated[method]['lm_judge_rating'],
                linestyle='--',
                linewidth=2,
                marker=marker,
                markersize=6,
                markerfacecolor=color,
                markeredgecolor='black',
                markeredgewidth=1,
                label=f'{method}',
                color=color)
    
    # Customize the plot
    plt.xlabel('Factor', fontsize=10, color='black')
    plt.ylabel('LM Judge Rating', fontsize=10, color='black')
    
    # Add finer x-axis ticks
    min_factor = min(min(aggregated[method]['factor']) for method in methods)
    max_factor = max(max(aggregated[method]['factor']) for method in methods)
    plt.xticks(np.arange(min_factor, max_factor + 0.5, 0.5), fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    
    # Add legend
    plt.legend(loc="lower right", 
              fontsize=5, 
              frameon=True, 
              fancybox=True, 
              framealpha=0.8, 
              shadow=True)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if write_to_path:
        plt.savefig(write_to_path / "lm_judge_rating.png", 
                   dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()