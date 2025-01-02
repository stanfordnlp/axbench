# script based tests for our released datasets

import pandas as pd

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
for dataset in datasets:
    test_train_latent_match(
        f"./axbench/{dataset}/prod_2b_l10_v1/generate/train_data.parquet", 
        f"./axbench/{dataset}/prod_2b_l10_v1/inference/latent_eval_data.parquet")
    test_train_latent_match(
        f"./axbench/{dataset}/prod_2b_l20_v1/generate/train_data.parquet", 
        f"./axbench/{dataset}/prod_2b_l20_v1/inference/latent_eval_data.parquet")
    test_train_latent_match(
        f"./axbench/{dataset}/prod_9b_it_l20_v1/generate/train_data.parquet", 
        f"./axbench/{dataset}/prod_9b_it_l20_v1/inference/latent_eval_data.parquet")
    test_train_latent_match(
        f"./axbench/{dataset}/prod_9b_it_l31_v1/generate/train_data.parquet",
        f"./axbench/{dataset}/prod_9b_it_l31_v1/inference/latent_eval_data.parquet")

print("==All tests passed!==")