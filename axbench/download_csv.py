import pandas as pd
import json
import textwrap

def convert_parquet_to_csv(parquet_path, csv_path):
    try:
        # Read the parquet file
        print(f"Reading parquet file from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        
        df.to_csv(csv_path, index=False)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Specify your input and output paths
    parquet_file = "/afs/cs.stanford.edu/u/qinanyu/axbench/axbench/output_llama/layer_20_concise/lsreft_200_noise_prompt_20/evaluate/steering_data.parquet"
    csv_file = "/afs/cs.stanford.edu/u/qinanyu/axbench/axbench/output_llama/layer_20_concise/lsreft_200_noise_prompt_20/evaluate/steering_data.csv"
    
    # Convert the file
    df = convert_parquet_to_csv(parquet_file, csv_file)