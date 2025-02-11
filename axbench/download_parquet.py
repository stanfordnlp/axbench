import pandas as pd
import json
import textwrap

def convert_parquet_to_csv(parquet_path, json_path):
    try:
        # Read the parquet file
        print(f"Reading parquet file from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        steered_dict = {}
        # Create nested dictionary with steered_input as keys
        for i, input_val in enumerate(df['steered_input']):
            if input_val not in steered_dict:
                steered_dict[input_val] = [{
                'concept_ratings': df['GemmaScopeSAE_LMJudgeEvaluator_relevance_concept_ratings'].iloc[i],
                'instruction_ratings': df['GemmaScopeSAE_LMJudgeEvaluator_relevance_instruction_ratings'].iloc[i],
                'fluency_ratings': df['GemmaScopeSAE_LMJudgeEvaluator_fluency_ratings'].iloc[i],
                'factor': df['factor'].iloc[i],
                'steered_generation': textwrap.fill(df['GemmaScopeSAE_steered_generation'].iloc[i], width=80)
            }]
            else:
                steered_dict[input_val].append({
                'concept_ratings': df['GemmaScopeSAE_LMJudgeEvaluator_relevance_concept_ratings'].iloc[i],
                'instruction_ratings': df['GemmaScopeSAE_LMJudgeEvaluator_relevance_instruction_ratings'].iloc[i],
                'fluency_ratings': df['GemmaScopeSAE_LMJudgeEvaluator_fluency_ratings'].iloc[i],
                'factor': df['factor'].iloc[i],
                'steered_generation': textwrap.fill(df['GemmaScopeSAE_steered_generation'].iloc[i], width=80)
            })

        
        
        print(f"Saving JSON file to: {json_path}")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(steered_dict, f, indent=4)
        
        print(f"Successfully saved dictionary to {json_path}")
        print(f"Number of entries: {len(steered_dict)}")
        
        return steered_dict
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Specify your input and output paths
    parquet_file = "output_llama/llama_test_suppress/inference.parquet"
    csv_file = "output_llama/llama_test_suppress/inference.parquet"
    
    # Convert the file
    df = convert_parquet_to_csv(parquet_file, csv_file)