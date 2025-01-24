# script based tests for our released datasets

import pandas as pd
import json
import torch

def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            concept, ref = data["concept"], data["ref"]
            concept_genres_map = data["concept_genres_map"][concept]
            ref = data["ref"]
            flatten_data = {
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": data["concept_id"]
            }
            metadata += [flatten_data]  # Return the metadata as is
    return metadata

def test_train_latent_match(train_path, latent_path):
    print(f"Testing {train_path} and {latent_path}")
    train_df = pd.read_parquet(train_path)
    latent_df = pd.read_parquet(latent_path)
    train_concepts = set(train_df.output_concept.unique())
    latent_concepts = set(latent_df.output_concept.unique())
    for c in train_concepts-{"EEEEE"}:
        assert c in latent_concepts
    concept_id_map = {}
    for i, row in train_df.iterrows():
        concept_id_map[row.concept_id] = row.output_concept
    for i, row in latent_df.iterrows():
        if row.category == "positive":
            assert row.output_concept == concept_id_map[row.concept_id]

datasets = ["concept10", "concept500"]
tasks = ["prod_2b_l10_v1", "prod_2b_l20_v1", "prod_9b_l20_v1", "prod_9b_l31_v1"]
for dataset in datasets:
    for task in tasks:
        test_train_latent_match(
            f"./axbench/{dataset}/{task}/generate/train_data.parquet", 
            f"./axbench/{dataset}/{task}/inference/latent_eval_data.parquet")

        train_df = pd.read_parquet(f"./axbench/{dataset}/{task}/generate/train_data.parquet")
        latent_df = pd.read_parquet(f"./axbench/{dataset}/{task}/inference/latent_eval_data.parquet")
        
        # checking negative examples count
        assert len(train_df[train_df.category == "negative"]) == (72 * 3)
        if dataset == "concept10":
            assert len(train_df[train_df.category == "positive"]) == (10 * 72)
        else:
            assert len(train_df[train_df.category == "positive"]) == (500 * 72)

        # checking genre distribution for negative examples
        train_df_negative = train_df[train_df.category == "negative"]
        genre_distribution = train_df_negative.concept_genre.value_counts()
        assert genre_distribution.loc["text"] == 72
        assert genre_distribution.loc["math"] == 72
        assert genre_distribution.loc["code"] == 72

        # checking genre distribution for positive examples
        latent_df_positive = latent_df[latent_df.category == "positive"]
        genre_distribution = latent_df_positive.concept_genre.value_counts()
        assert genre_distribution.loc["text"]/len(latent_df_positive) >= 0.5

        # checking concept in df is matching metadata
        metadata = load_metadata_flatten(f"./axbench/{dataset}/{task}/generate/metadata.jsonl")
        concept_id_map = {}
        for md in metadata:
            concept_id_map[md["concept_id"]] = md["concept"]
        for i, row in train_df.iterrows():
            if row.concept_id != -1:
                assert row.concept_id in concept_id_map
                assert row.output_concept == concept_id_map[row.concept_id]

        for i, row in latent_df.iterrows():
            if row.concept_id != -1 and "//" not in row.output_concept:
                assert row.concept_id in concept_id_map
                assert row.output_concept == concept_id_map[row.concept_id]

print("==All tests passed with Concept10 and Concept500 !==")  

datasets = ["concept16k"]
tasks = ["prod_2b_l20_v1", "prod_9b_l20_v1"]
for dataset in datasets:
    for task in tasks:
        train_df = pd.read_parquet(f"./axbench/{dataset}/{task}/generate/train_data.parquet")

        metadata = load_metadata_flatten(f"./axbench/{dataset}/{task}/generate/metadata.jsonl")
        concept_id_map = {}
        for md in metadata:
            concept_id_map[md["concept_id"]] = md["concept"]
        for i, row in train_df.iterrows():
            if row.concept_id != -1:
                assert row.concept_id in concept_id_map
                assert row.output_concept == concept_id_map[row.concept_id]

print("==All tests passed with Concept16k !==")  

# testing releasing models
MODEL_PATHS = [
    "./axbench/results/prod_2b_l20_concept16k_lsreft",
    "./axbench/results/prod_2b_l20_concept16k_diffmean",
    "./axbench/results/prod_9b_l20_concept16k_lsreft",
    "./axbench/results/prod_9b_l20_concept16k_diffmean",
]
for model_path in MODEL_PATHS:
    print(f"Testing {model_path}")
    metadata = load_metadata_flatten(f"{model_path}/train/metadata.jsonl")
    curr_id = 0
    for md in metadata:
        assert md["concept_id"] != curr_id
        curr_id += 1
