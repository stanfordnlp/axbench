import argparse
import pickle
from plotnine import (
    ggplot, aes, geom_line, theme_minimal, labs, scale_color_manual, scale_x_continuous,
    scale_y_continuous, geom_boxplot, geom_violin, geom_histogram, element_text, theme,
    facet_wrap, facet_grid, theme_gray, theme_set, ylim, geom_point, geom_density, geom_bin_2d,
    geom_area, geom_jitter, stat_summary, geom_tile, scale_fill_gradient2, geom_hline, geom_bar,
    position_dodge2, scale_x_log10, scale_y_log10, stat_count, element_blank, geom_path, arrow,
    coord_flip, geom_text, position_stack
)
import json
import pandas as pd
import os
import matplotlib.font_manager as fm
from matplotlib import rcParams
import itertools
import numpy as np
from scipy.stats import bootstrap, ttest_rel
from collections import defaultdict

from evaluate import data_generator
from axbench import LatentStatsEvaluator
from tqdm import tqdm

# set the theme
rcParams['font.family'] = "P052"
theme_set(theme_gray(base_family="P052") + theme(axis_text_x=element_text(angle=45, hjust=0.5)))


RESULTS_FOLDER = "/nlp/scr/wuzhengx/pyreax/axbench/results"
SUBFOLDERS = [
    "prod_2b_l10_concept500",
    "prod_2b_l20_concept500",
    "prod_9b_l20_concept500",
    "prod_9b_l31_concept500",
    # "prod_2b_l20_concept16k",
    # "prod_9b_l20_concept16k",
]
PLOT_FOLDER = "paper/"

METHODS = ["ig", "lsreft", "steering_vec", "sft", "probe", "no_grad", "crossfit", "lora", "loreft"]
METHOD_MAP = {
    "LsReFT": "ReFT-r1",
    "SteeringVector": "SSV",
    "SparseLinearProbe": "Probe-SL",
    "DiffMean": "DiffMean",
    "PCA": "PCA",
    "LAT": "LAT",
    "GemmaScopeSAE": "SAE",
    "IntegratedGradients": "IG",
    "InputXGradients": "IxG",
    "LinearProbe": "Probe",
    "PromptSteering": "Prompt",
    "LoReFT": "LoReFT",
    "LoRA": "LoRA",
    "SFT": "SFT",
}
MODEL_MAP = {
    "2b": "Gemma-2-2B",
    "9b": "Gemma-2-9B",
}
LAYER_MAP = {
    "l10": "L10",
    "l20": "L20",
    "l31": "L31",
}
METRIC_MAP = {
    "macro_avg_accuracy": "Acc. (macro-avg)",
    "max_act": "Max Act.",
    "optimal_threshold": "Optimal Thresh.",
    "roc_auc": "ROC AUC",
    "max_lm_judge_rating": "Overall Score",
    "max_fluency_rating": "Fluency Score",
    "max_relevance_concept_rating": "Concept Score",
    "max_relevance_instruction_rating": "Instruct Score",
    "max_factor": "Steering Factor",
    "overall_accuracy": "Overall Accuracy",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
}
FLOAT_METRICS = ['macro_avg_accuracy', 'max_act', 'optimal_threshold', 'roc_auc',
                 'max_lm_judge_rating', 'max_fluency_rating', 'max_relevance_concept_rating',
                 'max_relevance_instruction_rating', 'max_factor', 'overall_accuracy', 'f1', 'precision', 'recall']
INT_METRICS = ['concept_id']


def prettify_df(df):
    # rename columns
    for metric in FLOAT_METRICS:
        df[metric] = df[metric].astype(float)
    for metric in INT_METRICS:
        df[metric] = df[metric].astype(int)
    df["method"] = df["method"].apply(lambda x: METHOD_MAP.get(x, x))
    df = df.rename(columns={"method": "Method"})
    df.columns = [METRIC_MAP[col] if col in METRIC_MAP else col for col in df.columns]
    return df


def mean_and_ci(group, n_bootstraps=1000, ci=0.95):
    values = group["values"].values
    # Compute mean
    mean_value = np.mean(values)
    if np.isnan(mean_value):
        print(values)
        print("bruh its nan")
    # Compute bootstrap CI
    result = bootstrap(
        data=(values,),
        statistic=np.mean,
        n_resamples=n_bootstraps,
        confidence_level=ci,
        method="percentile"
    )
    lower_ci, upper_ci = result.confidence_interval
    return pd.Series({"mean": mean_value, "lower_ci": lower_ci, "upper_ci": upper_ci})


def format_ci(mean, lower_ci, upper_ci, only_mean=False, percent=False):
    result = ""
    if only_mean:
        if percent:
            result = f"{mean:.1%}"
        else:
            result = f"{mean:.3f}"
    else:
        if percent:
            result = f"{mean:.1%}$^{{+{upper_ci-mean:.3f}}}_{{-{mean-lower_ci:.3f}}}$"
        else:
            result = f"{mean:.3f}$^{{+{upper_ci-mean:.3f}}}_{{-{mean-lower_ci:.3f}}}$"
    result = result.replace("%", "\\%")
    return result


def agg_first_non_nan(vals):
    for val in vals:
        if isinstance(val, list) or np.all(val):
            return val
    return np.nan


def split_metric(metric):
    splitted = metric.split(".")
    if len(splitted) > 4:
        splitted[3] = '.'.join(splitted[3:])
    return tuple(splitted[:4])


def main(reload=False, pairs=False):
    # make plot folder if not exists
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    all_dfs = []
    if reload or not os.path.exists(f"{PLOT_FOLDER}/df.pkl"):
        for folder in SUBFOLDERS:
            parts = folder.split("_")
            assert len(parts) == 4
            _, model, layer, split = parts
            dfs = []
            for method in METHODS:
                method_folder = f"{RESULTS_FOLDER}/{folder}_{method}/evaluate"
                inference_folder = f"{RESULTS_FOLDER}/{folder}_{method}/inference"
                if not os.path.exists(method_folder):
                    continue
                    
                # load latent eval
                megadict = defaultdict(dict)
                latent_eval = f"{method_folder}/latent.jsonl"
                if os.path.exists(latent_eval):
                    print(f"Processing latent for {method}...")
                    with open(latent_eval, "r") as f:
                        for line in f:
                            json_line = json.loads(line)
                            megadict[json_line['concept_id']]['latent'] = json_line['results']

                # load latent parquet
                if not os.path.exists(f"{inference_folder}/latent_data.parquet"):
                    print(f"Skipping {method} because no latent.parquet found")
                else:
                    df_generator = data_generator(inference_folder, mode="latent")
                    for concept_id, df in tqdm(df_generator, total=500):
                        eval_results = {}
                        for method in METHOD_MAP:
                            if f"{method}_max_act" not in df.columns:
                                continue
                            evaluator = LatentStatsEvaluator(method)
                            eval_result = evaluator.compute_metrics(df)
                            eval_results[method] = eval_result
                        megadict[concept_id]['latent']['LatentStatsEvaluator'] = eval_results
                
                # load steering eval
                steering_eval = f"{method_folder}/steering.jsonl"
                if os.path.exists(steering_eval):
                    print(f"Processing steering for {method}...")
                    with open(steering_eval, "r") as f:
                        for line in f:
                            json_line = json.loads(line)
                            megadict[json_line['concept_id']]['steering'] = json_line['results']
                
                megalist = [{'concept_id': concept_id, **data} for concept_id, data in megadict.items()]
                df = pd.json_normalize(megalist)
                dfs.append(df)
        
            # merge dfs based on concept_id column, pick the first one
            if len(dfs) == 0:
                continue
            df = pd.concat(dfs)
            # print(list(df.columns))
            # print("Duplicates:", len(df.duplicated(subset=["concept_id", "model", "layer"])))
            df = df.groupby("concept_id").first().reset_index()
            df["model"] = MODEL_MAP[model]
            df["layer"] = LAYER_MAP[layer]
            df["split"] = split
            df["identifier"] = f"{model}, {layer}, {split}"
            all_dfs.append(df)

        # save df
        df = pd.concat(all_dfs)
        df.to_pickle(f"{PLOT_FOLDER}/df.pkl")
    else:
        df = pd.read_pickle(f"{PLOT_FOLDER}/df.pkl")
    
    # make each method a row
    id_vars = ['concept_id', 'model', 'layer', 'split', 'identifier']
    value_vars = [col for col in df.columns if col not in id_vars]
    melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='metric', value_name='value')
    melted_df["method"] = melted_df["metric"].apply(lambda x: x.split(".")[2])
    melted_df["metric"] = melted_df["metric"].apply(lambda x: ".".join(x.split(".")[3:]))
    pivot_df = melted_df.pivot_table(index=id_vars + ['method'], columns='metric', values='value', aggfunc=agg_first_non_nan).reset_index()

    # steering metrics
    no_factor = ["PromptSteering", "LoReFT", "LoRA", "SFT"]
    steering_metrics = ["lm_judge_rating", "fluency_ratings", "relevance_concept_ratings", "relevance_instruction_ratings"]
    for metric in steering_metrics:
        pivot_df[metric] = pivot_df.apply(lambda row: np.mean(row[metric]) if isinstance(row[metric], list) and row["method"] in no_factor else row[metric], axis=1)
    pivot_df["max_lm_judge_rating"] = pivot_df["lm_judge_rating"].apply(lambda x: max(x) if isinstance(x, list) else x)
    pivot_df["max_lm_judge_rating_idx"] = pivot_df["lm_judge_rating"].apply(lambda x: np.argmax(x) if isinstance(x, list) else 0)
    pivot_df["max_fluency_rating"] = pivot_df.apply(lambda row: row["fluency_ratings"][int(row["max_lm_judge_rating_idx"])] if isinstance(row["fluency_ratings"], list) else row["fluency_ratings"], axis=1)
    pivot_df["max_relevance_concept_rating"] = pivot_df.apply(lambda row: row["relevance_concept_ratings"][int(row["max_lm_judge_rating_idx"])] if isinstance(row["relevance_concept_ratings"], list) else row["relevance_concept_ratings"], axis=1)
    pivot_df["max_relevance_instruction_rating"] = pivot_df.apply(lambda row: row["relevance_instruction_ratings"][int(row["max_lm_judge_rating_idx"])] if isinstance(row["relevance_instruction_ratings"], list) else row["relevance_instruction_ratings"], axis=1)
    pivot_df["max_factor"] = pivot_df.apply(lambda row: row["factor"][int(row["max_lm_judge_rating_idx"])] if isinstance(row["factor"], list) else row["factor"], axis=1)

    # fix types
    pivot_df = prettify_df(pivot_df)
    no_factor = ["Prompt", "LoReFT", "LoRA", "SFT"]
    print(pivot_df.head())
    print("Columns:", sorted(list(pivot_df.columns)))
    print(pivot_df.iloc[0])

    # print duplicates
    dups = pivot_df[pivot_df.duplicated(subset=["identifier", "concept_id", "Method"])].sort_values(by=["identifier", "concept_id", "Method"])
    print("Duplicates:", len(dups))

    common_fpr = np.linspace(0, 1, 38)
    for split in df["split"].unique():
        df = pivot_df[pivot_df["split"] == split]
        if len(df) == 0:
            continue
        
        # plot precision vs recall
        df_subset = df.copy()
        df_subset = df_subset[["Method", "model", "layer", "identifier", "Precision", "Recall"]]
        df_subset = df_subset.dropna(subset=["Precision", "Recall"])
        df_subset = df_subset.groupby(["identifier", "Method", "model", "layer"]).mean().reset_index()
        plot = (
            ggplot(df_subset, aes(x="Precision", y="Recall", color="Method"))
            + facet_wrap('model + ": " + layer')
            + geom_point()
            + labs(x="Precision", y="Recall")
        )
        plot.save(f"{PLOT_FOLDER}/{split}_precision_vs_recall.pdf", width=4, height=3)

        # plot max_factor
        df_subset = df.copy()
        df_subset = df_subset.dropna(subset=["Steering Factor"])
        df_subset = df_subset[~df_subset["Method"].isin(no_factor + ["SSV"])]
        # df_subset = df_subset.groupby(["Method", "model", "layer", "Steering Factor"]).count().reset_index()
        print(df_subset)
        plot = (
            ggplot(df_subset, aes(y="Steering Factor", fill='model + ": " + layer'))
            + geom_boxplot(outlier_alpha=0.3)
            + facet_wrap("Method")
            # + geom_violin()
            # + stat_count(geom="point")
            + labs(x="Setting", y="Steering Factor", fill="")
            + scale_y_log10()
            + theme(legend_position="top", axis_text_x=element_blank(), axis_ticks_x=element_blank())
        )
        plot.save(f"{PLOT_FOLDER}/{split}_best_factor.pdf", width=6, height=4)

        def winrate(row):
            method = "SAE"
            prompt_equivalent = df[(df["Method"] == method) & (df["identifier"] == row["identifier"]) & (df["concept_id"] == row["concept_id"])].iloc[0]
            if not isinstance(row["Overall Score"], (float, int)) or not isinstance(prompt_equivalent["Overall Score"], (float, int)):
                return np.nan
            return 1.0 if row["Overall Score"] > prompt_equivalent["Overall Score"] else 0.0 if row["Overall Score"] < prompt_equivalent["Overall Score"] else 0.5
        
        df["Winrate"] = df.apply(lambda row: winrate(row), axis=1)

        # make latex table
        detection_order = []
        steering_order = []
        for metric in ["ROC AUC", "Overall Score", "Winrate", "Overall Accuracy", "F1", "Precision", "Recall"]:
            with open(f"{PLOT_FOLDER}/{split}_{metric}.txt", "w") as f:
                for only_mean in [True, False]:
                    df_subset: pd.DataFrame = df.copy().dropna(subset=[metric])
                    df_subset = df_subset.rename(columns={metric: "values"})[["Method", "identifier", "values"]]
                    # print count of nans
                    f.write("NaN count:\n")
                    for method in df_subset["Method"].unique():
                        f.write(f"{method}: {sum(df_subset[df_subset['Method'] == method]['values'].apply(lambda x: np.isnan(x)))}\n")
                    avg_df = df_subset.copy()
                    avg_df["identifier"] = "Average"
                    df_subset = pd.concat([df_subset, avg_df])
                    df_subset = df_subset.groupby(["identifier", "Method"]).apply(mean_and_ci).reset_index()
                    
                    # Append the averages to the original dataframe
                    df_subset["values"] = df_subset.apply(lambda row: format_ci(row['mean'], row['lower_ci'], row['upper_ci'], only_mean, percent=(metric == "Winrate")), axis=1)
                    df_subset = df_subset.pivot(index="Method", columns="identifier", values="values").reset_index()
                    df_subset = df_subset.sort_values(by="Average", ascending=False)

                    # df_subset_avg = df.copy()
                    # df_subset_avg = df_subset_avg.rename(columns={metric: "values"})
                    # df_subset_avg = df_subset_avg[["Method", "identifier", "values"]].groupby(["Method"]).apply(mean_and_ci).reset_index()
                    # df_subset_avg["values"] = df_subset_avg.apply(lambda row: format_ci(row['mean'], row['lower_ci'], row['upper_ci'], only_mean, percent=(metric == "Winrate")), axis=1)
                    # df_subset["Average"] = df_subset.apply(lambda row: df_subset_avg[df_subset_avg["Method"] == row["Method"]]["values"].values[0], axis=1)
                    # df_subset = df_subset.sort_values(by="Average", ascending=False)
                    if metric == "ROC AUC":
                        detection_order = df_subset["Method"].tolist()
                    elif metric == "Overall Score":
                        steering_order = df_subset["Method"].tolist()
                    elif metric in ["Winrate"]:
                        df_subset["Method"] = pd.Categorical(df_subset["Method"], categories=steering_order, ordered=True)
                    else:
                        df_subset["Method"] = pd.Categorical(df_subset["Method"], categories=detection_order, ordered=True)
                    flattened = df_subset.to_latex(index=False)
                    f.write(flattened)
                    f.write("\n\n")

                # do paired t test on Latent ROC AUC between each pair of methods
                df_subset = df.copy()[["Method", "identifier", "concept_id", metric]]
                df_subset = df_subset.pivot(index=["identifier", "concept_id"], columns="Method", values=metric).reset_index()
                for identifier in list(df_subset["identifier"].unique()) + [None]:
                    for method1, method2 in itertools.combinations(df["Method"].unique(), 2):
                        df_subset_t = df_subset[df_subset["identifier"] == identifier] if identifier is not None else df_subset
                        if method1 not in df_subset_t.columns or method2 not in df_subset_t.columns:
                            continue
                        roc_auc_1 = list(df_subset_t[method1])
                        roc_auc_2 = list(df_subset_t[method2])
                        # print all nans
                        method1_nan = sum(np.isnan(roc_auc_1))
                        method2_nan = sum(np.isnan(roc_auc_2))
                        if method1_nan > 0 or method2_nan > 0:
                            # print(f"{method1} vs {method2}: skipping: {method1_nan}, {method2_nan}")
                            continue
                        t_stat, p_value = ttest_rel(roc_auc_1, roc_auc_2, nan_policy="raise")
                        f.write(f"{identifier}: {method1} vs {method2}: T-statistic: {t_stat}, P-value: {p_value} {'(Significant)' if p_value < 0.05 else ''}\n")

        # plot roc auc vs max lm judge rating
        df_subset = df.copy()
        df_subset = df_subset[["Method", "model", "layer", "identifier", "ROC AUC", "Overall Score"]]
        df_subset2 = df_subset.dropna(subset=["ROC AUC", "Overall Score"])
        df_subset2 = df_subset2.groupby(["identifier", "Method", "model", "layer"]).mean().reset_index()
        print(df_subset2)
        plot = (
            ggplot(df_subset2, aes(x="ROC AUC", y="Overall Score", color="Method"))
            + facet_grid("~model + layer")
            + geom_point()
            + labs(x="ROC AUC", y="Overall Score")
        )
        df_subset3 = df_subset[df_subset["Method"].isin(["Prompt", "LoReFT", "LoRA", "SFT"])].groupby(["identifier", "Method", "model", "layer"]).mean().reset_index()
        plot += geom_hline(df_subset3, aes(color="Method", yintercept="Overall Score"), linetype="dashed")
        plot.save(f"{PLOT_FOLDER}/{split}_summary.pdf", width=8, height=5)

        # plot each steering metric
        df_subset = df.copy()[["Method", "model", "layer", "identifier", "Concept Score", "Instruct Score", "Fluency Score"]]
        df_subset = df_subset.dropna(subset=["Concept Score", "Instruct Score", "Fluency Score"])
        df_subset = df_subset.melt(id_vars=["Method", "model", "layer", "identifier"], value_vars=["Concept Score", "Instruct Score", "Fluency Score"], var_name="metric", value_name="value")
        df_subset["Method"] = pd.Categorical(df_subset["Method"], categories=steering_order)
        plot = (
            ggplot(df_subset, aes(x="Method", y="value", fill='model + ": " + layer'))
            + facet_grid("metric~")
            # + facet_grid("model + layer ~")
            # + stat_summary(fun_data="mean_cl_boot", geom="bar", position="dodge")
            + geom_boxplot(outlier_shape="", position=position_dodge2(preserve="single"))
            # + geom_violin(position=position_dodge2(preserve="single"))
            + labs(fill="", y="Score")
            + theme(legend_position="top", axis_text_x=element_text(angle=0, hjust=0.5))
        )
        plot.save(f"{PLOT_FOLDER}/{split}_steering.pdf", width=8, height=4)

        # plot winrates
        df_subset = df.copy()
        df_subset = df_subset.dropna(subset=["Overall Score"])
        methods = df_subset["Method"].unique()
        df_subset = df_subset.pivot(index=id_vars, columns="Method", values="Overall Score").reset_index()
        print(df_subset)
        def winrate(row, method):
            if not isinstance(row[method], (float, int)) or not isinstance(row["Prompt"], (float, int)):
                return np.nan
            return 1.0 if row[method] > row["Prompt"] else 0.0 if row[method] < row["Prompt"] else 0.5
        for method in methods:
            df_subset[method] = df_subset.apply(lambda row: winrate(row, method), axis=1)
        df_subset = df_subset.melt(id_vars=id_vars, value_vars=methods, var_name="Method", value_name="Winrate")
        df_subset["Count"] = 1
        df_subset = df_subset.groupby(["Method", "identifier", "model", "layer", "split"]).sum().reset_index()
        df_subset["Method"] = pd.Categorical(df_subset["Method"], categories=steering_order)
        df_subset["Winrate"] = df_subset["Winrate"] / df_subset["Count"]
        df_subset["Text"] = df_subset["Winrate"].apply(lambda x: f"{x:.1%}")
        df_subset = df_subset.dropna()
        plot = (
            ggplot(df_subset, aes(y="Winrate", x="Method"))
            + facet_wrap('model + ": " + layer', scales="free_x")
            + geom_bar(stat="identity", position="stack")
            + geom_text(aes(label="Text", x="Method", y="Winrate"), position=position_stack(vjust=1.0), nudge_y=0.08)
            + labs(x="Method", y="Winrate")
            + coord_flip()
            + ylim(0, 1)
        )
        plot.save(f"{PLOT_FOLDER}/{split}_winrate.pdf", width=8, height=5)

        # plot factor vs score
        factor_df = df.copy()
        factor_df = factor_df.dropna(subset=["factor"])
        factor_df = factor_df[~factor_df["Method"].isin(no_factor)]
        factors = [val for val in factor_df["factor"]]
        factor_dict = {
            "Steering Factor": [x for val in factor_df["factor"] for x in val],
            "Overall Score": [x for val in factor_df["lm_judge_rating"] for x in val],
            "Concept Score": [x for val in factor_df["relevance_concept_ratings"] for x in val],
            "Instruct Score": [x for val in factor_df["relevance_instruction_ratings"] for x in val],
            "Fluency Score": [x for val in factor_df["fluency_ratings"] for x in val],
        }
        lens = [len(val) for val in factors]
        for col in id_vars + ["Method"]:
            factor_dict[col] = [val for i, val in enumerate(factor_df[col]) for _ in range(lens[i])]
        factor_df_og = pd.DataFrame(factor_dict)
        factor_df_og["Method"] = pd.Categorical(factor_df_og["Method"], categories=[x for x in steering_order if x not in no_factor])
        factor_df = factor_df_og.melt(id_vars=id_vars + ["Method", "Steering Factor"], value_vars=["Overall Score", "Concept Score", "Instruct Score", "Fluency Score"], var_name="metric", value_name="value")
        plot = (
            ggplot(factor_df, aes(x="Steering Factor", y="value", color="Method"))
            + stat_summary(fun_data="mean_cl_boot", geom="line")
            + scale_x_log10()
            + facet_grid("metric~model + layer")
            + labs(x="Steering Factor", y="Overall Score")
        )
        plot.save(f"{PLOT_FOLDER}/{split}_steering_factor.pdf", width=8, height=6)

        # concept score vs instruct score for each method
        factor_df_subset = factor_df_og[["Method", "model", "layer", "identifier", "Steering Factor", "Concept Score", "Instruct Score", "Fluency Score", "Overall Score"]]
        factor_df_subset = factor_df_subset.groupby(["Method", "model", "layer", "identifier", "Steering Factor"]).mean().reset_index()
        factor_df_subset = factor_df_subset.dropna(subset=["Steering Factor", "Concept Score", "Instruct Score", "Fluency Score", "Overall Score"])
        factor_df_subset = factor_df_subset.sort_values(by="Steering Factor").reset_index()
        plot = (
            ggplot(factor_df_subset, aes(x="Concept Score", y="Instruct Score"))
            + facet_wrap('model + ": " + layer')
            + geom_path(aes(color="Method", group="Method"), arrow=arrow(type="closed", length=0.05))
            # + geom_point(aes(color="Method"))
        )
        plot.save(f"{PLOT_FOLDER}/{split}_concept_vs_instruct.pdf", width=4, height=3)

        # make a new df from the lists roc_curve.fpr and roc_curve.tpr
        roc_df = df.copy().dropna(subset=["roc_curve.fpr", "roc_curve.tpr"])
        fprs = [val for val in roc_df["roc_curve.fpr"]]
        tprs = [val for val in roc_df["roc_curve.tpr"]]
        for i in range(len(fprs)):
            tprs[i] = [0.0] + list(np.interp(common_fpr, fprs[i], tprs[i]))
            fprs[i] = [0.0] + list(common_fpr)
        roc_dict = {
            "fpr": [x for val in fprs for x in val],
            "tpr": [x for val in tprs for x in val],
        }
        # for key in ["fpr", "tpr"]:
        #     for val in roc_df[f"roc_curve.{key}"]:
        #         vals = []
        #         for x in val:
        #             if len(vals) != 0:
        #                 if key == "fpr": vals.append(x)
        #                 else: vals.append(vals[-1])
        #             vals.append(x)
        #         roc_dict[key].extend(vals)
        lens = [len(val) for val in fprs]
        for col in id_vars + ["Method"]:
            roc_dict[col] = [val for i, val in enumerate(roc_df[col]) for _ in range(lens[i])]
        roc_df = pd.DataFrame(roc_dict)
        roc_mean_df = roc_df.groupby(['model', 'layer', 'split', 'identifier', 'Method', 'fpr']).mean().reset_index()
        roc_mean_df["Method"] = pd.Categorical(roc_mean_df["Method"], categories=detection_order)
        roc_df["Method"] = pd.Categorical(roc_df["Method"], categories=detection_order)
        print("Main ROC DF", len(roc_df))
        print("Mean ROC DF", len(roc_mean_df))

        plot = (
            ggplot(roc_mean_df, aes(x="fpr", y="tpr", color="Method", group="Method"))
            + facet_wrap('model + ": " + layer')
            + geom_line()
            + labs(x="False Positive Rate", y="True Positive Rate")
        )
        plot.save(f"{PLOT_FOLDER}/{split}_roc_mean.pdf", width=4, height=3)

        plot = (
            ggplot(roc_df[(roc_df["model"] == "Gemma-2-2B") & (roc_df["layer"] == "L10")], aes(x="fpr", y="tpr", color="Method", group="concept_id"))
            + facet_grid("model + layer ~ Method")
            + geom_line(alpha=0.1)
            # + stat_summary(fun_data="mean_cl_boot", geom="line", color="black")
            + labs(x="False Positive Rate", y="True Positive Rate")
            + theme(legend_position="none", axis_text_x=element_text(angle=90, hjust=0.5))
        )
        # Add average ROC curve to the plot
        plot += geom_line(roc_mean_df[(roc_mean_df["model"] == "Gemma-2-2B") & (roc_mean_df["layer"] == "L10")], aes(x="fpr", y="tpr"), color="black", size=1.0, alpha=0.5)
        plot.save(f"{PLOT_FOLDER}/{split}_roc.pdf", width=8, height=1.75)

        plot = (
            ggplot(roc_df, aes(x="fpr", y="tpr", color="Method", group="concept_id"))
            + facet_grid("model + layer ~ Method")
            + geom_line(alpha=0.1)
            # + stat_summary(fun_data="mean_cl_boot", geom="line", color="black")
            + labs(x="False Positive Rate", y="True Positive Rate")
            + theme(legend_position="none")
        )
        # Add average ROC curve to the plot
        plot += geom_line(roc_mean_df, aes(x="fpr", y="tpr"), color="black", size=1.0, alpha=0.5)
        plot.save(f"{PLOT_FOLDER}/{split}_roc_all.pdf", width=8, height=5)

        # make a plot for each metric
        for metric in FLOAT_METRICS:
            metric_name = METRIC_MAP[metric]
            plot = (
                ggplot(
                    df.dropna(subset=[metric_name]),
                    aes(x="Method", y=metric_name, fill="layer")
                )
                + facet_grid("~model")
                + geom_boxplot(outlier_alpha=0.3)
                + labs(x="Method", y=metric_name)
                + ylim(min(0, df[metric_name].min()), max(1, df[metric_name].max()))
            )
            plot.save(f"{PLOT_FOLDER}/{split}_{metric}.pdf", width=8, height=5)

        # Make scatter plots comparing each pair of float metrics
        if pairs:
            for metric1, metric2 in itertools.combinations(FLOAT_METRICS, 2):
                metric1_name = METRIC_MAP[metric1]
                metric2_name = METRIC_MAP[metric2]
                
                plot = (
                    ggplot(
                        df.dropna(subset=[metric1_name, metric2_name]),
                        aes(x=metric1_name, y=metric2_name, color="layer")
                    )
                    + facet_grid("identifier ~ Method")
                    + geom_point(alpha=0.5)
                    + labs(x=metric1_name, y=metric2_name)
                    + theme(legend_position="right")
                )
                plot.save(f"{PLOT_FOLDER}/{split}_{metric1}_vs_{metric2}.pdf", width=12, height=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--pairs", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
