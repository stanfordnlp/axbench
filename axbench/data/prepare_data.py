"""Download pyvene/axbench-concept500 from HuggingFace and prepare it
for concept detection inference (LatentQA, Activation Oracles, PromptDetection).

Creates the directory structure expected by inference.py:
  {dump_dir}/generate/metadata.jsonl
  {dump_dir}/train/config.json
  {dump_dir}/inference/latent_eval_data.parquet   (for overwrite_inference_data_dir)

IMPORTANT: metadata.jsonl is built from the parquet's own concept descriptions
(output_concept column), NOT from the neuronpedia JSON. The HF dataset contains
a specific curated set of 500 concepts whose IDs don't correspond to the first
500 entries of the neuronpedia JSON.
"""
import os, json, argparse
import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--hf_subdir", type=str, default="2b/l20",
                        help="Subdirectory in pyvene/axbench-concept500 (e.g. 2b/l20, 9b/l31)")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer number for train/config.json")
    parser.add_argument("--component", type=str, default="res",
                        help="Component for train/config.json")
    args = parser.parse_args()

    dump_dir = args.dump_dir
    os.makedirs(f"{dump_dir}/generate", exist_ok=True)
    os.makedirs(f"{dump_dir}/train", exist_ok=True)
    os.makedirs(f"{dump_dir}/inference", exist_ok=True)

    # 1. Download parquet files from HF
    print(f"Downloading parquet files from pyvene/axbench-concept500/{args.hf_subdir}...")
    test_path = hf_hub_download(
        repo_id="pyvene/axbench-concept500",
        filename=f"{args.hf_subdir}/test/data.parquet",
        repo_type="dataset",
    )
    train_path = hf_hub_download(
        repo_id="pyvene/axbench-concept500",
        filename=f"{args.hf_subdir}/train/data.parquet",
        repo_type="dataset",
    )
    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path)
    print(f"Loaded test: {len(test_df)} rows, train: {len(train_df)} rows")

    # 2. Build metadata.jsonl from the parquet's actual concepts
    # This ensures concept_id -> concept description alignment
    print("Building metadata.jsonl from parquet concepts...")
    concept_map = {}
    for _, row in test_df.drop_duplicates("concept_id").iterrows():
        cid = int(row["concept_id"])
        concept_map[cid] = {
            "concept": row["output_concept"],
            "genre": row.get("concept_genre", "text"),
            "sae_link": row.get("sae_link", ""),
        }

    metadata_path = f"{dump_dir}/generate/metadata.jsonl"
    with open(metadata_path, "w") as f:
        for concept_id in sorted(concept_map.keys()):
            info = concept_map[concept_id]
            entry = {
                "concept_id": concept_id,
                "concept": info["concept"],
                "ref": info["sae_link"],
                "concept_genres_map": {info["concept"]: [info["genre"]]},
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(concept_map)} concepts to {metadata_path}")

    # 3. Create config.json
    config = {"layer": args.layer, "component": args.component}
    config_path = f"{dump_dir}/train/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    print(f"Wrote config to {config_path}")

    # 4. Save test split as latent_eval_data.parquet
    eval_df = test_df.copy()
    if "sae_id" not in eval_df.columns:
        eval_df["sae_id"] = eval_df["concept_id"]
    eval_path = f"{dump_dir}/inference/latent_eval_data.parquet"
    eval_df.to_parquet(eval_path, index=False)
    print(f"Wrote {len(eval_df)} rows to {eval_path}")

    # 5. Save train data
    if "sae_id" not in train_df.columns:
        train_df["sae_id"] = train_df["concept_id"]
    train_out = f"{dump_dir}/generate/train_data.parquet"
    train_df.to_parquet(train_out, index=False)
    print(f"Wrote {len(train_df)} rows to {train_out}")

    print("Data preparation complete!")


if __name__ == "__main__":
    main()
