#!/bin/bash

# Set variables
MODEL="deepset/gbert-large"
MODELNAME="gbert-large"
DATA_FILE="data/AmDi.jsonl"
EMB_FILE="embeddings/AmDi.gbert-large.h5"
OUTPUT_DIR="results/${MODELNAME}"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# 1. Generate lexemes list dynamically from dataset.jsonl
LEXEMES=($(cat "$DATA_FILE" | jq -r '.lexem' | sort | uniq))

# 2. Run drift analysis for each lexeme
for LEXEME in "${LEXEMES[@]}"; do
  echo "Running drift analysis for lexeme: $LEXEME"
  python -m scripts.analyze_drift -d "$DATA_FILE" -l "$LEXEME" --emb_h5 "$EMB_FILE" --pairwise_test --output_dir "$OUTPUT_DIR/lexemes" --exclude_timespans 1995-2014 2006-2023
done

# 3. Aggregate all results
echo "Aggregating drift analysis results..."
python -m scripts.drift_summary_tool --input_dir "$OUTPUT_DIR/lexemes" --output_dir "$OUTPUT_DIR/drift_summaries" --alpha 0.05

# 4. Create visualizations
echo "Creating visualizations..."
python -m scripts.visualize_drift_summary_extended --summary_dir "$OUTPUT_DIR/drift_summaries" --output_dir "$OUTPUT_DIR/drift_summaries"

echo "Pipeline completed. Results are saved in $OUTPUT_DIR."

