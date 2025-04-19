import pandas as pd
import argparse
import os
import json

def prepare_truthfulqa_dataset(input_file, output_file):
    """
    Prepare the TruthfulQA dataset for evaluation.
    Ensures the dataset has the required columns:
    - question
    - truthful_answer
    - false_answer
    """
    print(f"Loading dataset from {input_file}")
    
    # Read the input CSV
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError("Input file must be CSV or JSON")
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # For your specific dataset format
    if 'Question' in df.columns and 'Correct Answers' in df.columns and 'Incorrect Answers' in df.columns:
        print("Found matching columns in the provided format")
        
        # Create the prepared dataset
        prepared_df = pd.DataFrame({
            'question': df['Question'],
            'truthful_answer': df['Correct Answers'],
            'false_answer': df['Incorrect Answers']
        })
        
        # Verify the dataset is valid
        # Remove any rows with NaN values
        original_count = len(prepared_df)
        prepared_df = prepared_df.dropna()
        if len(prepared_df) < original_count:
            print(f"Warning: Removed {original_count - len(prepared_df)} rows with missing values")
        
        # Make sure we have at least some rows
        if len(prepared_df) == 0:
            raise ValueError("No valid rows in the dataset after preparation")
        
        # Save the prepared dataset
        prepared_df.to_csv(output_file, index=False)
        print(f"Saved prepared dataset with {len(prepared_df)} rows to {output_file}")
        
        # Print a few examples
        print("\nExample data:")
        for i, row in prepared_df.head(3).iterrows():
            print(f"Example {i+1}:")
            print(f"  Question: {row['question']}")
            print(f"  Truthful answer: {row['truthful_answer']}")
            print(f"  False answer: {row['false_answer']}")
            print()
        
        return prepared_df
    else:
        raise ValueError("Your dataset doesn't have the expected columns. Please ensure it has 'Question', 'Correct Answers', and 'Incorrect Answers' columns.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare TruthfulQA dataset for evaluation')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input dataset file (CSV or JSON)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the prepared dataset (CSV)')
    
    args = parser.parse_args()
    
    prepare_truthfulqa_dataset(args.input_file, args.output_file)
