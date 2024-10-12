#!/bin/bash

# Define the start and end layer numbers
START=0  # Replace with the desired start layer
END=25   # Replace with the desired end layer

# Base URL for the file download
BASE_URL="https://neuronpedia-exports.s3.amazonaws.com/explanations-only/gemma-2-"

# Loop over the range of layers from START to END
for (( i=START; i<=END; i++ )); do
    # Construct the URLs for both resolutions for the current layer
    FILE_URL_16K="${BASE_URL}2b_${i}-gemmascope-res-16k.json"
    FILE_URL_65K="${BASE_URL}2b_${i}-gemmascope-res-65k.json"
    
    # Download the file for 16k resolution
    echo "Downloading ${FILE_URL_16K}..."
    curl -O "${FILE_URL_16K}"
    
    # Download the file for 65k resolution
    echo "Downloading ${FILE_URL_65K}..."
    curl -O "${FILE_URL_65K}"
done

echo "Download completed."