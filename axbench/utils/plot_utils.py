import os, glob, random, json, wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.metrics import auc
from matplotlib.colors import hsv_to_rgb
import seaborn as sns
import pandas as pd
from pathlib import Path
from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_wrap, geom_bar, geom_abline, xlim, scale_fill_manual,
    geom_text, position_dodge, ylim, labs, theme_bw, theme, element_text, scale_color_manual, coord_flip
)


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


def plot_aggregated_roc(jsonl_data, write_to_path=None, report_to=[], wandb_name=None):
    # Collect ROC data for each model
    metrics_list = [aggregated_result["results"]["AUCROCEvaluator"] 
                    for aggregated_result in jsonl_data]
    
    # Define common FPR thresholds for interpolation
    common_fpr = np.linspace(0, 1, 100)
    
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
    
    # Prepare data for plotting
    plot_data = []
    for model_name in tprs.keys():
        mean_tpr = np.mean(tprs[model_name], axis=0)
        mean_auc = np.mean(aucs[model_name])
        for fpr, tpr in zip(common_fpr, mean_tpr):
            plot_data.append({
                'FPR': fpr,
                'TPR': tpr,
                'Model': f"{model_name} (AUC = {mean_auc:.2f})"
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    p = (
        ggplot(df, aes(x='FPR', y='TPR', color='Model')) +
        geom_line(size=1) +
        geom_abline(slope=1, intercept=0, linetype='dashed', color='gray') +
        theme_bw() +
        labs(x='False Positive Rate (FPR)', y='True Positive Rate (TPR)') +
        theme(
            figure_size=(4, 4),
            legend_title=element_text(size=8),
            legend_text=element_text(size=6),
            axis_title=element_text(size=10),
            axis_text=element_text(size=8),
            plot_title=element_text(size=12),
            legend_position='right'
        )
    )
    
    # Optional: Customize colors if needed
    # COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', ...]  # Define your color palette
    # p += scale_color_manual(values=COLORS)
    
    # Save or show the plot
    if write_to_path:
        p.save(filename=str(write_to_path / "aggregated_roc.png"), dpi=300, bbox_inches='tight')
    else:
        print(p)

    # Report to wandb if wandb_name is provided
    if report_to is not None and "wandb" in report_to:
        # Prepare data for wandb.plot.line_series
        xs = common_fpr.tolist()
        ys = [np.mean(tprs[model], axis=0).tolist() for model in tprs]
        keys = [f"{model} (AUC = {np.mean(aucs[model]):.2f})" for model in tprs]
        wandb.log({"latent/roc_curve" : wandb.plot.line_series(
            xs=xs,
            ys=ys,
            keys=keys,
            title='Aggregated ROC Curve',
            xname='False Positive Rate (FPR)',
        )})


def plot_metrics(jsonl_data, configs, write_to_path=None, report_to=[], wandb_name=None):
    # Collect data into a list
    data = []
    for config in configs:
        evaluator_name = config['evaluator_name']
        metric_name = config['metric_name']
        y_label = config['y_label']
        use_log_scale = config['use_log_scale']
        
        for entry in jsonl_data:
            results = entry.get('results', {}).get(evaluator_name, {})
            for method, res in results.items():
                factors = res.get('factor', [])
                metrics = res.get(metric_name, [])
                # Ensure factors and metrics are lists
                if not isinstance(factors, list):
                    factors = [factors]
                if not isinstance(metrics, list):
                    metrics = [metrics]
                for f, m in zip(factors, metrics):
                    data.append({
                        'Factor': f,
                        'Value': m,
                        'Method': method,
                        'Metric': y_label,
                        'UseLogScale': use_log_scale
                    })

    # Create DataFrame and average metrics
    df = pd.DataFrame(data)
    df = df.groupby(['Method', 'Factor', 'Metric', 'UseLogScale'], as_index=False).mean()

    # Apply log transformation if needed
    df['TransformedValue'] = df.apply(
        lambda row: np.log10(row['Value']) if row['UseLogScale'] else row['Value'],
        axis=1
    )

    # Create the plot
    p = (
        ggplot(df, aes(x='Factor', y='TransformedValue', color='Method', group='Method')) +
        geom_line() +
        geom_point() +
        theme_bw() +
        labs(x='Factor', y='Value') +
        facet_wrap('~ Metric', scales='free_y', nrow=1) +  # Plots in a row
        theme(
            subplots_adjust={'wspace': 0.1},
            figure_size=(1.5 * len(configs), 3),  # Wider for more plots, taller height
            legend_position='right',
            legend_title=element_text(size=4),
            legend_text=element_text(size=6),
            axis_title=element_text(size=6),
            axis_text=element_text(size=6),
            axis_text_x=element_text(rotation=90, hjust=1),  # Rotate x-axis labels
            strip_text=element_text(size=6)
        )
    )

    # Save or show the plot
    if write_to_path:
        p.save(filename=str(write_to_path / "steering_plot.png"), dpi=300, bbox_inches='tight')
    else:
        print(p)

    # Report to wandb if wandb_name is provided
    if report_to is not None and "wandb" in report_to:
        # Separate data by metrics to prepare for wandb line series plotting
        line_series_plots = {}
        for metric in df['Metric'].unique():
            metric_data = df[df['Metric'] == metric]
            
            xs = metric_data['Factor'].unique().tolist()
            ys = [metric_data[metric_data['Method'] == method]['TransformedValue'].tolist() for method in metric_data['Method'].unique()]
            keys = [f"{method}" for method in metric_data['Method'].unique()]
            
            line_series_plots[f"steering/{metric}"] = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=keys,
                title=f"{metric}",
                xname='Factor'
            )
        wandb.log(line_series_plots)


def plot_accuracy_bars(jsonl_data, evaluator_name, write_to_path=None, report_to=[], wandb_name=None):
    
    # Get unique methods and sort them
    methods = set()
    for entry in jsonl_data:
        methods.update(entry['results'][evaluator_name].keys())
    methods = sorted(list(methods))
    
    # Initialize data structures for both metrics
    seen_accuracies = {method: [] for method in methods}
    unseen_accuracies = {method: [] for method in methods}
    
    # Collect data from all concepts
    for entry in jsonl_data:
        results = entry['results'][evaluator_name]
        for method in methods:
            if method in results:
                seen_accuracies[method].append(
                    results[method].get('hard_negative_seen_accuracy', 0))
                unseen_accuracies[method].append(
                    results[method].get('hard_negative_unseen_accuracy', 0))
    
    # Calculate means
    seen_means = {method: np.mean(vals) for method, vals in seen_accuracies.items()}
    unseen_means = {method: np.mean(vals) for method, vals in unseen_accuracies.items()}
    
    # Prepare data for plotting
    data = []
    for method in methods:
        data.append({'Method': method, 'AccuracyType': 'Seen', 'Accuracy': seen_means[method]})
        data.append({'Method': method, 'AccuracyType': 'Unseen', 'Accuracy': unseen_means[method]})
    
    df = pd.DataFrame(data)
    
    # Create the plot
    p = (
        ggplot(df, aes(x='Method', y='Accuracy', fill='AccuracyType')) +
        geom_bar(stat='identity', position=position_dodge(width=0.8), width=0.7) +
        geom_text(
            aes(label='round(Accuracy, 2)'),
            position=position_dodge(width=0.8),
            va='bottom',
            size=8,
            format_string='{:.2f}'
        ) +
        ylim(0, 1) +  # Set y-axis limits from 0 to 1
        theme_bw() +
        labs(x='Method', y='Accuracy') +
        theme(
            figure_size=(5, 2),
            legend_position='right',
            legend_title=element_text(size=5),
            legend_text=element_text(size=5),
            axis_title=element_text(size=5),
            axis_text=element_text(size=5),
            plot_title=element_text(size=5)
        )
    )

    # Save or show the plot
    if write_to_path:
        p.save(filename=str(write_to_path / "hard_negative_accuracy.png"), dpi=300, bbox_inches='tight')
    else:
        print(p)

    if report_to is not None and "wandb" in report_to:
        wandb.log({"latent/hard_negative_accuracy": wandb.Image(str(write_to_path / "hard_negative_accuracy.png"))})


def plot_win_rates(jsonl_data, write_to_path=None, report_to=[], wandb_name=None):    
    # Collect methods and baseline models
    methods = set()
    baseline_models = set()
    for entry in jsonl_data:
        winrate_results = entry.get('results', {}).get('WinRateEvaluator', {})
        for method_name, res in winrate_results.items():
            methods.add(method_name)
            baseline_models.add(res.get('baseline_model', 'Unknown'))
    methods = sorted(list(methods))
    baseline_models = sorted(list(baseline_models))
    
    # Assuming all methods are compared against the same baseline
    if len(baseline_models) == 1:
        baseline_model = baseline_models[0]
    else:
        # Handle multiple baselines if necessary
        baseline_model = baseline_models[0]  # For now, take the first one
    
    # Add the baseline method to methods if not already present
    if baseline_model not in methods:
        methods.append(baseline_model)
    
    # Initialize data structures
    win_rates = {method: [] for method in methods}
    loss_rates = {method: [] for method in methods}
    tie_rates = {method: [] for method in methods}
    
    # Collect data from all concepts
    num_concepts = len(jsonl_data)
    for entry in jsonl_data:
        winrate_results = entry.get('results', {}).get('WinRateEvaluator', {})
        for method in methods:
            if method == baseline_model:
                continue  # Handle baseline separately
            if method in winrate_results:
                res = winrate_results[method]
                win_rates[method].append(res.get('win_rate', 0) * 100)
                loss_rates[method].append(res.get('loss_rate', 0) * 100)
                tie_rates[method].append(res.get('tie_rate', 0) * 100)
            else:
                # If method is not present in this concept, assume zero rates
                win_rates[method].append(0.0)
                loss_rates[method].append(0.0)
                tie_rates[method].append(0.0)
    
    # For the baseline method, set win_rate=50%, loss_rate=50%, tie_rate=0%
    win_rates[baseline_model] = [50.0] * num_concepts
    loss_rates[baseline_model] = [50.0] * num_concepts
    tie_rates[baseline_model] = [0.0] * num_concepts
    
    # Calculate mean percentages
    win_means = {method: np.mean(vals) for method, vals in win_rates.items()}
    loss_means = {method: np.mean(vals) for method, vals in loss_rates.items()}
    tie_means = {method: np.mean(vals) for method, vals in tie_rates.items()}
    
    # Sort methods: baseline at top, then methods by descending win rate
    non_baseline_methods = [m for m in methods if m != baseline_model]
    sorted_methods = sorted(
        non_baseline_methods,
        key=lambda m: win_means[m],
        reverse=True
    )
    
    # Prepare data for plotting
    data = []
    for method in sorted_methods:
        data.append({'Method': method, 'Outcome': 'Win', 'Percentage': win_means[method]})
        data.append({'Method': method, 'Outcome': 'Tie', 'Percentage': tie_means[method]})
        data.append({'Method': method, 'Outcome': 'Loss', 'Percentage': loss_means[method]})
    
    df = pd.DataFrame(data)
    
    # Set the order of Outcome to control stacking order
    df['Outcome'] = pd.Categorical(df['Outcome'], categories=['Loss', 'Tie', 'Win'], ordered=True)
    # Reverse the methods list for coord_flip to display baseline at the top
    df['Method'] = pd.Categorical(df['Method'], categories=sorted_methods[::-1], ordered=True)
    
    # Create the plot
    p = (
        ggplot(df, aes(x='Method', y='Percentage', fill='Outcome')) +
        geom_bar(stat='identity', position='stack', width=0.8) +
        coord_flip() +  # Flip coordinates for horizontal bars
        theme_bw() +
        labs(
            y='Percentage (%)',
            x=''
        ) +
        theme(
            axis_text_x=element_text(size=4),
            axis_text_y=element_text(size=4),
            axis_title=element_text(size=4),
            legend_title=element_text(size=4),
            legend_text=element_text(size=4),
            figure_size=(3, len(sorted_methods) * 0.2 + 0.3)
        ) +
        scale_fill_manual(
            values={'Win': '#a6cee3', 'Tie': '#bdbdbd', 'Loss': '#fbb4ae'},
            guide='legend',
            name='Outcome'
        )
    )
    
    # Save or show the plot
    if write_to_path:
        p.save(filename=str(write_to_path / "winrate_plot.png"), dpi=300, bbox_inches='tight')
    else:
        print(p)

    if report_to is not None and "wandb" in report_to:
        wandb.log({"steering/winrate_plot": wandb.Image(str(write_to_path / "winrate_plot.png"))})
