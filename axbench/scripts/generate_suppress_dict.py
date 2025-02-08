import pandas as pd
from pprint import pprint
import json


df = pd.read_parquet('../output_llama/layer_20_concise/generate/train_data.parquet')

# Filter out 'EEEEE' and get unique pairs of concept and concept_id
filtered_df = df[df['output_concept'] != 'EEEEE']
concept_id_pairs = filtered_df[['output_concept', 'concept_id']].drop_duplicates()

# Create a dictionary with unique concepts and their IDs
suppress_dict = {row['output_concept']: row['concept_id'] for _, row in concept_id_pairs.iterrows()}

df_steering = pd.read_parquet('../output_llama/layer_20_concise/lsreft_20__0.005_0.3/evaluate/steering_data.parquet')
# Get unique pairs of steering prompts and input concepts
unique_pairs = df_steering[['steering_prompt_original', 'input_concept']].drop_duplicates()
steering_dict = {row['input_concept']: row['steering_prompt_original'] 
                for _, row in unique_pairs.iterrows()}

concept_suppress_dict = {}
for key, value in suppress_dict.items():
    concept_suppress_dict[key] = steering_dict[key]

# Write concept_suppress_dict to a JSON file
with open('../output_llama/layer_20_concise/concept_suppress_dict.json', 'w') as f:
    json.dump(concept_suppress_dict, f, indent=4)

