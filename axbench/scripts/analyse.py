import argparse
import pickle
from plotnine import (
    ggplot, aes, geom_line, theme_minimal, labs, scale_color_manual, scale_x_continuous,
    scale_y_continuous, geom_boxplot, geom_violin, geom_histogram, element_text, theme,
    facet_wrap, facet_grid, theme_gray, theme_set, ylim, geom_point, geom_density, geom_bin_2d,
    geom_area, geom_jitter, stat_summary,
)
import json
import pandas as pd
import os
import matplotlib.font_manager as fm
from matplotlib import rcParams
import itertools
import numpy as np
from scipy.stats import bootstrap, ttest_rel

# set the theme
rcParams['font.family'] = "P052"
theme_set(theme_gray(base_family="P052") + theme(axis_text_x=element_text(angle=45, hjust=0.5)))


RESULTS_FOLDER = "/nlp/scr/wuzhengx/pyreax/axbench/results"
SUBFOLDERS = [
    "prod_2b_l10_concept500",
    "prod_2b_l20_concept500",
    "prod_9b_l20_concept500",
    "prod_9b_l31_concept500",
    "prod_2b_l20_concept16k",
    "prod_9b_l20_concept16k",
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
    "macro_avg_accuracy": "Latent Acc. (macro-avg)",
    "max_act": "Latent Max Act.",
    "optimal_threshold": "Latent Optimal Thresh.",
    "roc_auc": "Latent ROC AUC",
}
FLOAT_METRICS = ['macro_avg_accuracy', 'max_act', 'optimal_threshold', 'roc_auc']
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
    # Compute bootstrap CI
    result = bootstrap(
        data=(values,),
        statistic=np.mean,
        n_resamples=n_bootstraps,
        confidence_level=ci,
        method="percentile"
    )
    lower_ci, upper_ci = result.confidence_interval
    plus_minus = (upper_ci - lower_ci) / 2
    return pd.Series({"mean": mean_value, "lower_ci": lower_ci, "upper_ci": upper_ci, "plus_minus": plus_minus})


def format_ci(mean, lower_ci, upper_ci, only_mean=False):
    if only_mean:
        return f"{mean:.3f}"
    return f"{mean:.3f}$^{{+{upper_ci-mean:.3f}}}_{{-{mean-lower_ci:.3f}}}$"


def main():
    # make plot folder if not exists
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    all_dfs = []
    for folder in SUBFOLDERS:
        parts = folder.split("_")
        assert len(parts) == 4
        _, model, layer, split = parts
        dfs = []
        for method in METHODS:
            method_folder = f"{RESULTS_FOLDER}/{folder}_{method}/evaluate"
            if not os.path.exists(method_folder):
                continue
            print(f"Processing {method}...")

            # load latent eval
            latent_eval = f"{method_folder}/latent.jsonl"
            data = []
            with open(latent_eval, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.json_normalize(data)
            dfs.append(df)
    
        # merge dfs based on concept_id column, pick the first one
        if len(dfs) == 0:
            continue
        df = pd.concat(dfs)
        df = df.groupby("concept_id").first().reset_index()
        df["model"] = MODEL_MAP[model]
        df["layer"] = LAYER_MAP[layer]
        df["split"] = split
        df["identifier"] = f"{model}, {layer}, {split}"
        all_dfs.append(df)
    
    df = pd.concat(all_dfs)
    
    # make each method a row
    id_vars = ['concept_id', 'model', 'layer', 'split', 'identifier']
    value_vars = [col for col in df.columns if col not in id_vars]
    melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='metric', value_name='value')
    melted_df[['evaluator', 'method', 'metric']] = melted_df['metric'].str.extract(r'results\.(\w+)\.(\w+)\.(.+)')
    pivot_df = melted_df.pivot_table(index=id_vars + ['method'], columns='metric', values='value', aggfunc='first').reset_index()
    pivot_df.columns.name = None
    pivot_df.columns = [col if isinstance(col, str) else col[1] for col in pivot_df.columns]

    # fix types
    pivot_df = prettify_df(pivot_df)
    print(pivot_df.head())
    print(list(pivot_df.columns))

    # print duplicates
    print("Duplicates:")
    print(pivot_df[pivot_df.duplicated(subset=["identifier", "concept_id", "Method"])][["identifier", "concept_id", "Method"]])

    common_fpr = np.linspace(0, 1, 38)
    for split in df["split"].unique():
        df = pivot_df[pivot_df["split"] == split]
        if len(df) == 0:
            continue

        # make latex table
        for metric in ["Latent Acc. (macro-avg)", "Latent ROC AUC"]:
            for only_mean in [True, False]:
                df_subset: pd.DataFrame = df.copy()
                df_subset = df_subset.rename(columns={metric: "values"})[["Method", "identifier", "values"]]
                # print count of nans
                for method in df_subset["Method"].unique():
                    print(f"{method}: {sum(df_subset[df_subset['Method'] == method]['values'].isna())}")
                print(df_subset.head())
                input()
                df_subset = df_subset.groupby(["identifier", "Method"]).apply(mean_and_ci).reset_index()
                df_subset["values"] = df_subset.apply(lambda row: format_ci(row['mean'], row['lower_ci'], row['upper_ci'], only_mean), axis=1)
                df_subset = df_subset.pivot(index="Method", columns="identifier", values="values").reset_index()

                df_subset_avg = df.copy()
                df_subset_avg = df_subset_avg.rename(columns={metric: "values"})
                df_subset_avg = df_subset_avg[["Method", "identifier", "values"]].groupby(["Method"]).apply(mean_and_ci).reset_index()
                df_subset_avg["values"] = df_subset_avg.apply(lambda row: format_ci(row['mean'], row['lower_ci'], row['upper_ci'], only_mean), axis=1)
                df_subset["Average"] = df_subset_avg["values"]
                df_subset = df_subset.sort_values(by="Average", ascending=False)
                flattened = df_subset.to_latex(index=False)
                print(flattened)
            
            # do paired t test on Latent ROC AUC between each pair of methods
            df_subset = df.copy()[["Method", "identifier", "concept_id", metric]]
            df_subset = df_subset.pivot(index=["identifier", "concept_id"], columns="Method", values=metric).reset_index()
            print(df_subset.head())
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
                    print(f"{identifier}: {method1} vs {method2}: T-statistic: {t_stat}, P-value: {p_value} {'(Significant)' if p_value < 0.05 else ''}")

        # make a plot for each metric
        for metric in FLOAT_METRICS:
            metric_name = METRIC_MAP[metric]
            plot = (
                ggplot(
                    df,
                    aes(x="Method", y=metric_name, fill="layer")
                )
                + facet_grid("~model")
                + geom_boxplot(outlier_alpha=0.3)
                + labs(x="Method", y=metric_name)
                + ylim(min(0, df[metric_name].min()), max(1, df[metric_name].max()))
            )
            plot.save(f"{PLOT_FOLDER}/{split}_{metric}.pdf", width=10, height=5)

        # make a new df from the lists roc_curve.fpr and roc_curve.tpr
        fprs = [val for val in df["roc_curve.fpr"]]
        tprs = [val for val in df["roc_curve.tpr"]]
        for i in range(len(fprs)):
            tprs[i] = [0.0] + list(np.interp(common_fpr, fprs[i], tprs[i]))
            fprs[i] = [0.0] + list(common_fpr)
        roc_dict = {
            "fpr": [x for val in fprs for x in val],
            "tpr": [x for val in tprs for x in val],
        }
        # for key in ["fpr", "tpr"]:
        #     for val in df[f"roc_curve.{key}"]:
        #         vals = []
        #         for x in val:
        #             if len(vals) != 0:
        #                 if key == "fpr": vals.append(x)
        #                 else: vals.append(vals[-1])
        #             vals.append(x)
        #         roc_dict[key].extend(vals)
        lens = [len(val) for val in fprs]
        for col in id_vars + ["Method"]:
            roc_dict[col] = [val for i, val in enumerate(df[col]) for _ in range(lens[i])]
        roc_df = pd.DataFrame(roc_dict)
        roc_mean_df = roc_df.groupby(['model', 'layer', 'split', 'identifier', 'Method', 'fpr']).mean().reset_index()
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
            ggplot(roc_df, aes(x="fpr", y="tpr", color="Method", group="concept_id"))
            + facet_grid("model + layer ~ Method")
            + geom_line(alpha=0.2)
            # + stat_summary(fun_data="mean_cl_boot", geom="line", color="black")
            + labs(x="False Positive Rate", y="True Positive Rate")
        )
        # Add average ROC curve to the plot
        plot += geom_line(roc_mean_df, aes(x="fpr", y="tpr"), color="black", size=1.5, linetype="dashed")
        plot.save(f"{PLOT_FOLDER}/{split}_roc.pdf", width=10, height=5)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--folder", type=str, default="prod_9b_l20_concept500")
    # args = parser.parse_args()
    # main(**vars(args))
    main()