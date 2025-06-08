import argparse
import warnings

import pandas as pd
import torch
from evaluate import load
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import create_instruction_dataset, load_and_combine_data

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a finetuned model for Korean text deobfuscation.")
    parser.add_argument("--base_model_name", type=str, default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B", help="Base pretrained model name.")
    parser.add_argument("--model_dir", type=str, default="./hyperclova-deobfuscation-lora", help="Directory containing the finetuned LoRA model.")
    parser.add_argument("--data_dir", type=str, default="./", help="Directory containing the data files for evaluation.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to use for evaluation.")
    parser.add_argument("--qualitative_samples", type=int, default=10, help="Number of samples for qualitative review.")
    return parser.parse_args()


def generate_deobfuscated_text(model, tokenizer, obfuscated_text: str, max_length: int = 256) -> str:
    """Generate deobfuscated text from obfuscated input."""
    prompt = f"### 지시사항:\n다음 난독화된 한국어 텍스트를 원래 텍스트로 복원해주세요.\n\n난독화된 텍스트: {obfuscated_text}\n\n### 응답:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### 응답:" in response:
        return response.split("### 응답:")[1].strip()
    return response


def calculate_metrics(predictions, references):
    """Calculate BLEU and ROUGE scores."""
    bleu_metric = load("bleu")
    rouge_metric = load("rouge")

    bleu_score = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)

    return {
        'bleu': bleu_score['bleu'],
        'rouge1': rouge_score['rouge1'],
        'rouge2': rouge_score['rouge2'],
        'rougeL': rouge_score['rougeL']
    }


def main():
    args = get_args()

    # --- 1. Device and Model Loading ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    print(f"\nLoading base model: {args.base_model_name}")

    model_kwargs = {"device_map": "auto"}
    if device.type != 'cuda':
        # MPS/CPU 환경에서는 bfloat16으로 로드합니다.
        model_kwargs["torch_dtype"] = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        **model_kwargs
    )

    print(f"Loading LoRA adapter from: {args.model_dir}")
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model.eval()
    print("Model and tokenizer loaded successfully.")

    # --- 2. Data Loading ---
    combined_df = load_and_combine_data(args.data_dir)
    # We use the full dataset here, and then sample for evaluation
    instruction_df = create_instruction_dataset(combined_df, sample_size=None)
    
    if args.num_samples > len(instruction_df):
        print(f"Warning: Requested {args.num_samples} samples, but dataset only has {len(instruction_df)}. Using all samples.")
        args.num_samples = len(instruction_df)
        
    eval_samples = instruction_df.sample(n=args.num_samples, random_state=42)
    print(f"\nEvaluating on {len(eval_samples)} samples...")

    # --- 3. Quantitative Evaluation ---
    predictions = []
    references = []

    for idx, row in eval_samples.iterrows():
        obfuscated = row['input'].split("난독화된 텍스트: ")[1]
        original = row['output']
        predicted = generate_deobfuscated_text(model, tokenizer, obfuscated)
        predictions.append(predicted)
        references.append(original)

        if len(predictions) % 20 == 0:
            print(f"  - Generated {len(predictions)}/{len(eval_samples)} predictions...")

    metrics = calculate_metrics(predictions, references)
    print("\n=== Quantitative Evaluation Results ===")
    print(f"BLEU Score: {metrics['bleu']:.4f}")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")

    # --- 4. Qualitative Evaluation ---
    print("\n=== Qualitative Evaluation Results ===")
    qualitative_df = eval_samples.sample(n=min(args.qualitative_samples, len(eval_samples)), random_state=42)
    
    for idx, row in qualitative_df.iterrows():
        obfuscated = row['input'].split("난독화된 텍스트: ")[1]
        original = row['output']
        # Find the prediction we already generated
        predicted = predictions[eval_samples.index.get_loc(idx)]

        print(f"\n--- Sample {idx} (Category: {row['category']}) ---")
        print(f"Obfuscated: {obfuscated}")
        print(f"Original:   {original}")
        print(f"Predicted:  {predicted}")


if __name__ == "__main__":
    main() 