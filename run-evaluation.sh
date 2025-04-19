#!/bin/bash
# Script to run the complete evaluation process for Alif model on Urdu TruthfulQA

# Default values
INPUT_DATASET=""
OUTPUT_DIR="./results"
MODEL_ID="large-traversaal/Alif-1.0-8B-Instruct"
NUM_SAMPLES=-1
SENTENCE_MODEL="paraphrase-multilingual-MiniLM-L12-v2"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input_dataset)
      INPUT_DATASET="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model_id)
      MODEL_ID="$2"
      shift 2
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --sentence_model)
      SENTENCE_MODEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if input dataset is provided
if [ -z "$INPUT_DATASET" ]; then
  echo "Error: Input dataset is required"
  echo "Usage: $0 --input_dataset path/to/dataset.csv [--output_dir results_dir] [--model_id model_name] [--num_samples count]"
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Install dependencies
echo "Installing required dependencies..."
pip install -q transformers bitsandbytes rouge_score nltk torch evaluate pandas numpy tqdm sentence-transformers

# Prepare dataset
echo "Preparing the dataset..."
PREPARED_DATASET="$OUTPUT_DIR/prepared_dataset.csv"
python truthfulqa-data-preparation.py --input_file "$INPUT_DATASET" --output_file "$PREPARED_DATASET"

if [ $? -ne 0 ]; then
  echo "Error: Failed to prepare the dataset. Exiting."
  exit 1
fi

# Run evaluation
echo "Running evaluation with Alif model..."
python alif-truthfulqa-evaluation.py \
  --dataset_path "$PREPARED_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --model_id "$MODEL_ID" \
  --num_samples "$NUM_SAMPLES" \
  --sentence_model "$SENTENCE_MODEL"

if [ $? -ne 0 ]; then
  echo "Error: Evaluation failed. Check logs for details."
  exit 1
fi

echo "Evaluation completed successfully!"
echo "Results saved in: $OUTPUT_DIR"
echo "Summary metrics available in: $OUTPUT_DIR/metrics.json"
echo "Full results available in: $OUTPUT_DIR/evaluation_results.csv"

# Print summary metrics
echo "Evaluation Metrics Summary:"
cat "$OUTPUT_DIR/metrics.json" | python -m json.tool
