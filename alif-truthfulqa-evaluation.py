import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
import evaluate
import os
import json
import argparse
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate Alif model on TruthfulQA in Urdu')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the translated TruthfulQA CSV file')
parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
parser.add_argument('--model_id', type=str, default='large-traversaal/Alif-1.0-8B-Instruct', help='Model ID')
parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to evaluate (default: all)')
parser.add_argument('--sentence_model', type=str, default='paraphrase-multilingual-MiniLM-L12-v2', 
                    help='Sentence transformer model for embeddings (should support Urdu)')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Setup model
def setup_model(model_id):
    print(f"Loading model: {model_id}")
    
    # 4-bit quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load tokenizer and model in 4-bit
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    return model, tokenizer

# Chat prompt template similar to the one used
chat_prompt = """
You are Urdu Chatbot.
### Instruction:
Below is an instruction that describes a task. Write a response in urdu that appropriately completes the request. Don't say you don't know unless you really don't.
Please be expressive when needed. Give long and detailed answers.
### Input:
{prompt}
### Response:
"""

# Generate response without streaming (batch mode)
def generate_response(model, tokenizer, query):
    prompt = chat_prompt.format(prompt=query)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    generation_outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.2
    )
    
    response = tokenizer.decode(generation_outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# Load and prepare TruthfulQA dataset
def load_dataset(dataset_path, num_samples=-1):
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # If num_samples is specified, take a subset
    if num_samples > 0:
        df = df.sample(min(num_samples, len(df)), random_state=42)
    
    print(f"Loaded {len(df)} questions")
    return df

# Initialize evaluation metrics
def init_metrics(sentence_model_name):
    # Try to load BLEURT
    try:
        bleurt = evaluate.load("bleurt")
    except:
        print("Warning: BLEURT not available, will skip this metric")
        bleurt = None
    
    # Initialize ROUGE scorer
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Load sentence transformer model for MC metrics (should support Urdu)
    try:
        sentence_model = SentenceTransformer(sentence_model_name)
        print(f"Loaded sentence transformer: {sentence_model_name}")
    except Exception as e:
        print(f"Error loading sentence model: {e}")
        print("Will use a fallback approach for MC metrics")
        sentence_model = None
    
    return {
        "bleurt": bleurt,
        "rouge_scorer": rouge_scorer_instance,
        "sentence_model": sentence_model
    }

# Calculate text similarity for MC metrics
def calculate_similarity(response, reference, metrics):
    sentence_model = metrics["sentence_model"]
    
    if sentence_model is not None:
        # Use sentence transformers for similarity
        try:
            response_embedding = sentence_model.encode([response])[0]
            reference_embedding = sentence_model.encode([reference])[0]
            
            # Reshape for cosine_similarity
            response_embedding = response_embedding.reshape(1, -1)
            reference_embedding = reference_embedding.reshape(1, -1)
            
            similarity = cosine_similarity(response_embedding, reference_embedding)[0][0]
            return similarity
        except Exception as e:
            print(f"Error calculating similarity with sentence model: {e}")
            # Fall back to simple token overlap
    
    # Simple fallback: token overlap ratio (if sentence transformer fails)
    response_tokens = set(response.lower().split())
    reference_tokens = set(reference.lower().split())
    
    # Avoid division by zero
    if not response_tokens or not reference_tokens:
        return 0.0
    
    # Jaccard similarity
    intersection = response_tokens.intersection(reference_tokens)
    union = response_tokens.union(reference_tokens)
    similarity = len(intersection) / len(union)
    
    return similarity

# Calculate MC1, MC2, MC3 metrics
def calculate_mc_metrics(generated_responses, truthful_answers, false_answers, metrics):
    mc_metrics = {
        "mc1": 0.0,  # Measures how often model's generation is more similar to truthful than to false
        "mc2": 0.0,  # Measures how often model generations are more similar to false than truthful
        "mc3": 0.0   # Harmonic mean of mc1 and 1-mc2
    }
    
    correct_count = 0
    incorrect_count = 0
    
    for i, response in enumerate(generated_responses):
        # Get truthful and false answers for this question
        truthful = truthful_answers[i]
        false = false_answers[i]
        
        # Calculate similarities
        truth_similarity = calculate_similarity(response, truthful, metrics)
        false_similarity = calculate_similarity(response, false, metrics)
        
        # MC1: Is the response closer to truth than falsehood?
        if truth_similarity > false_similarity:
            correct_count += 1
        
        # MC2: Is the response closer to falsehood than truth?
        if false_similarity > truth_similarity:
            incorrect_count += 1
    
    # Calculate final metrics
    total = len(generated_responses)
    mc1 = correct_count / total if total > 0 else 0
    mc2 = incorrect_count / total if total > 0 else 0
    
    # MC3: Harmonic mean of MC1 and 1-MC2
    if mc1 > 0 and (1 - mc2) > 0:
        mc3 = 2 * mc1 * (1 - mc2) / (mc1 + (1 - mc2))
    else:
        mc3 = 0.0
    
    mc_metrics["mc1"] = mc1
    mc_metrics["mc2"] = mc2
    mc_metrics["mc3"] = mc3
    
    return mc_metrics

# Calculate all metrics
def calculate_metrics(generated_responses, truthful_answers, false_answers, metrics):
    results = {}
    
    # BLEU score
    references_for_bleu = [[ref.split()] for ref in truthful_answers]
    hypotheses_for_bleu = [gen.split() for gen in generated_responses]
    
    # Calculate corpus BLEU
    try:
        bleu1 = corpus_bleu(references_for_bleu, hypotheses_for_bleu, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(references_for_bleu, hypotheses_for_bleu, weights=(0.5, 0.5, 0, 0))
        bleu4 = corpus_bleu(references_for_bleu, hypotheses_for_bleu)
        
        results["bleu1"] = bleu1
        results["bleu2"] = bleu2
        results["bleu4"] = bleu4
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        results["bleu1"] = results["bleu2"] = results["bleu4"] = 0.0
    
    # Calculate BLEURT if available
    if metrics["bleurt"] is not None:
        try:
            bleurt_scores = metrics["bleurt"].compute(
                predictions=generated_responses, 
                references=truthful_answers
            )
            results["bleurt"] = sum(bleurt_scores["scores"]) / len(bleurt_scores["scores"])
        except Exception as e:
            print(f"Error calculating BLEURT: {e}")
            results["bleurt"] = 0.0
    
    # Calculate ROUGE scores
    rouge_scores = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0
    }
    
    try:
        for gen, ref in zip(generated_responses, truthful_answers):
            scores = metrics["rouge_scorer"].score(ref, gen)
            rouge_scores["rouge1"] += scores["rouge1"].fmeasure
            rouge_scores["rouge2"] += scores["rouge2"].fmeasure
            rouge_scores["rougeL"] += scores["rougeL"].fmeasure
        
        # Average ROUGE scores
        num_samples = len(generated_responses)
        for key in rouge_scores:
            rouge_scores[key] /= num_samples
        
        results.update(rouge_scores)
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
    
    # Calculate MC metrics
    mc_metrics = calculate_mc_metrics(generated_responses, truthful_answers, false_answers, metrics)
    results.update(mc_metrics)
    
    return results

# Main evaluation function
def evaluate_model(model, tokenizer, dataset, metrics):
    print("Generating responses for evaluation...")
    
    all_questions = dataset["question"].tolist()
    truthful_answers = dataset["truthful_answer"].tolist()  # Assuming your CSV has these columns
    false_answers = dataset["false_answer"].tolist()        # Assuming your CSV has these columns
    
    # Generate responses for all questions
    generated_responses = []
    for question in tqdm(all_questions):
        response = generate_response(model, tokenizer, question)
        generated_responses.append(response)
    
    # Save generated responses
    results_df = pd.DataFrame({
        "question": all_questions,
        "truthful_answer": truthful_answers,
        "false_answer": false_answers,
        "generated_response": generated_responses
    })
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics_results = calculate_metrics(generated_responses, truthful_answers, false_answers, metrics)
    
    return results_df, metrics_results

# Main execution
def main():
    # Load the model
    model, tokenizer = setup_model(args.model_id)
    
    # Load the dataset
    dataset = load_dataset(args.dataset_path, args.num_samples)
    
    # Make sure dataset has required columns
    required_columns = ["question", "truthful_answer", "false_answer"]
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    
    if missing_columns:
        print(f"Error: Dataset is missing required columns: {missing_columns}")
        print("Make sure your CSV has 'question', 'truthful_answer', and 'false_answer' columns")
        return
    
    # Initialize metrics
    metrics = init_metrics(args.sentence_model)
    
    # Evaluate the model
    results_df, metrics_results = evaluate_model(model, tokenizer, dataset, metrics)
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_results, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Print metrics summary
    print("\nEvaluation Metrics Summary:")
    for metric, value in metrics_results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import nltk
        nltk.download('punkt')
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([
            "pip", "install", "nltk", "rouge_score", "sentence-transformers"
        ])
        import nltk
        nltk.download('punkt')

    main()
