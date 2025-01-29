import pandas as pd
import json

def parse_feature_description_to_json(df, model, layer, component, width):
    all_raw_df = df[(df["Layer"] == layer) & (df["Sae Type"] == component) & (df["Sae Size"] == width)]
    high_quality_df = all_raw_df[
        (all_raw_df["Ensemble Concat (All) Input Success"] == True) & 
        (all_raw_df["Ensemble Concat (All) Output Success"] == True)
    ].reset_index(drop=True)
    
    output_json = []
    for index, row in high_quality_df.iterrows():
        output_json += [
            {
                "modelId":model,
                "layer":f"{layer}-gemmascope-{component}-{width}",
                "index":row["Feature"],
                "description":row["Ensemble Raw (All) Description"],
                "explanationModelName":"FeatureDescription",
                "typeName":"oai_token-act-pair"
            }
        ]
    with open(f'{model}_{layer}-featured-{component}-{width}.json', 'w') as f:
        json.dump(output_json, f)

df = pd.read_csv("./Feature-Descriptions/descriptions/gemma-2-2b.csv")
parse_feature_description_to_json(df, "gemma-2-2b", 10, "res", "16k")

df = pd.read_csv("./Feature-Descriptions/descriptions/gemma-2-2b.csv")
parse_feature_description_to_json(df, "gemma-2-2b", 20, "res", "16k")