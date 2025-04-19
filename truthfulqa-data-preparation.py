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
    
    # Map TruthfulQA columns to our required format
    column_mapping = {}
    
    # Try to identify the question column
    question_columns = ['question', 'Question', 'urdu_question', 'question_urdu', 'translated_question']
    for col in question_columns:
        if col in df.columns:
            column_mapping['question'] = col
            break
    
    # Try to identify the truthful answer column
    truthful_columns = ['truthful_answer', 'correct_answer', 'answer', 'true_answer', 
                       'urdu_truthful_answer', 'truthful_answer_urdu', 'translated_truthful_answer']
    for col in truthful_columns:
        if col in df.columns:
            column_mapping['truthful_answer'] = col
            break
    
    # Try to identify false answer column
    false_columns = ['false_answer', 'incorrect_answer', 'wrong_answer', 
                    'urdu_false_answer', 'false_answer_urdu', 'translated_false_answer']
    for col in false_columns:
        if col in df.columns:
            column_mapping['false_answer'] = col
            break
    
    # Check if we found all required columns
    missing_columns = []
    for required_col in ['question', 'truthful_answer', 'false_answer']:
        if required_col not in column_mapping:
            missing_columns.append(required_col)
    
    if missing_columns:
        print(f"Warning: Could not identify columns for: {missing_columns}")
        print("Available columns:", df.columns.tolist())
        
        # Try best-effort approach to create missing columns
        
        # If we have question but missing truthful/false answers
        if 'question' in column_mapping and 'truthful_answer' not in column_mapping:
            if 'correct_answers' in df.columns:
                # Handle TruthfulQA format with lists
                print("Using 'correct_answers' as truthful_answer (taking first answer)")
                df['truthful_answer'] = df['correct_answers'].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x)
                )
                column_mapping['truthful_answer'] = 'truthful_answer'
        
        if 'question' in column_mapping and 'false_answer' not in column_mapping:
            if 'incorrect_answers' in df.columns:
                # Handle TruthfulQA format with lists
                print("Using 'incorrect_answers' as false_answer (taking first answer)")
                df['false_answer'] = df['incorrect_answers'].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x)
                )
                column_mapping['false_answer'] = 'false_answer'
    
    # Check again if we're still missing columns
    missing_columns = []
    for required_col in ['question', 'truthful_answer', 'false_answer']:
        if required_col not in column_mapping:
            missing_columns.append(required_col)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Please ensure your dataset has these columns or equivalents.")
    
    # Create the prepared dataset
    prepared_df = pd.DataFrame({
        'question': df[column_mapping['question']],
        'truthful_answer': df[column_mapping['truthful_answer']],
        'false_answer': df[column_mapping['false_answer']]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare TruthfulQA dataset for evaluation')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input dataset file (CSV or JSON)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the prepared dataset (CSV)')
    
    args = parser.parse_args()
    
    prepare_truthfulqa_dataset(args.input_file, args.output_file)
