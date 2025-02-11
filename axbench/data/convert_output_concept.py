import pandas as pd
import json

def read_gemma_csv():
    # Read the CSV file
    df = pd.read_csv('llama-3.1-8b.csv')
    # Filter out 'mlp' entries from 'SAE Type'
    df = df[df['Sae Type'] != 'mlp']
    return df

def convert_to_format(row, layer):
    return {
        "modelId": "llama3.1-8b",
        "layer": f"{layer}-llamascope-res-32k",
        "index": str(row['Feature']),
        "description": row['Ensemble Raw (All) Description'],
        "explanationModelName": "gpt-4o-mini",
        "typeName": "oai_token-act-pair"
    }

def process_layers():
    # Read the filtered data
    df = read_gemma_csv()
    
    # Get unique layers
    unique_layers = df['Layer'].unique().tolist()
    
    # List to store all JSON files
    json_files = []
    
    # Process each layer
    for layer in unique_layers:
        # Filter data for current layer
        layer_df = df[df['Layer'] == layer]       
        # Convert each row to the new format
        formatted_data = [convert_to_format(row, layer) for _, row in layer_df.iterrows()]
        # Create JSON filename
        filename = f'output_llama/output_llama_layer_{layer}_data.json'       
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump(formatted_data, f, indent=2)       
        # Add to list of JSON files
        json_files.append(filename)
    
    return unique_layers, json_files

if __name__ == "__main__":
    # Process the data and get results
    unique_layers, json_files = process_layers()
    
    # Print results
    print("Unique layers:", unique_layers)
    print("Generated JSON files:", json_files)
