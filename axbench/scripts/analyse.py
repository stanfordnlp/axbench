import argparse
import pickle
from plotnine import (
    ggplot, aes, geom_line, theme_minimal, labs, scale_color_manual, scale_x_continuous,
    scale_y_continuous, geom_boxplot, geom_violin, geom_histogram, element_text, theme,
    facet_wrap, facet_grid, theme_gray, theme_set, ylim,
)
import json
import pandas as pd
import os
import matplotlib.font_manager as fm
from matplotlib import rcParams
import itertools

# set the theme
rcParams['font.family'] = "P052"
theme_set(theme_gray(base_family="P052") + theme(axis_text_x=element_text(angle=45, hjust=1)))


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

METHODS = ["ig", "lsreft", "steering_vec", "sft", "probe", "no_grad", "crossfit"]
METHOD_MAP = {
    "LsReFT": "ReFT-r1",
    "SteeringVector": "SSV",
    "SparseLinearProbe": "Probe-SL",
    "DiffMean": "DiffMean",
    "PCA": "PCA",
    "LAT": "LAT",
    "GemmaScopeSAE": "SAE",
}
MODEL_MAP = {
    "2b": "Gemma 2B",
    "9b": "Gemma 9B",
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
    "roc_auc": "Latent AUC",
}
FLOAT_METRICS = ['macro_avg_accuracy', 'max_act', 'optimal_threshold', 'roc_auc']


def prettify_df(df):
    # rename columns
    for metric in FLOAT_METRICS:
        df[metric] = df[metric].astype(float)
    df["method"] = df["method"].map(METHOD_MAP)
    df.columns = [METRIC_MAP[col] if col in METRIC_MAP else col for col in df.columns]
    df = df.dropna()
    return df


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
    # make method a column
    id_vars = ['concept_id', 'model', 'layer', 'split', 'identifier']
    value_vars = [col for col in df.columns if col not in id_vars]

    # Melt the DataFrame to have a long format
    melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='metric', value_name='value')

    # Extract the method and metric from the 'metric' column
    melted_df[['evaluator', 'method', 'metric']] = melted_df['metric'].str.extract(r'results\.(\w+)\.(\w+)\.(.+)')

    # Pivot the DataFrame to have methods as rows and metrics as columns
    pivot_df = melted_df.pivot_table(index=id_vars + ['method'], columns='metric', values='value', aggfunc='first').reset_index()

    # Flatten the columns
    pivot_df.columns.name = None
    pivot_df.columns = [col if isinstance(col, str) else col[1] for col in pivot_df.columns]

    # fix types
    pivot_df = prettify_df(pivot_df)
    print(pivot_df.head())
    print(list(pivot_df.columns))

    # make smaller df for roc_auc
    # sort by roc_auc
    for split in df["split"].unique():
        pivot_df_split = pivot_df[pivot_df["split"] == split]
        for metric in FLOAT_METRICS:
            metric_name = METRIC_MAP[metric]
            plot = (
                ggplot(
                    pivot_df_split,
                    aes(x="method", y=metric_name, fill="layer")
                )
                + facet_grid("~model")
                + geom_boxplot(outlier_alpha=0.3)
                + labs(x="Method", y=metric_name)
                + ylim(min(0, pivot_df_split[metric_name].min()), max(1, pivot_df_split[metric_name].max()))
            )
            plot.save(f"{PLOT_FOLDER}/{split}_{metric}.pdf", width=10, height=5)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--folder", type=str, default="prod_9b_l20_concept500")
    # args = parser.parse_args()
    # main(**vars(args))
    main()