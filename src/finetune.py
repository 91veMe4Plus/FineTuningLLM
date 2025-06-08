import argparse
import warnings

import torch
from datasets import Dataset, DatasetDict
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

from utils import create_instruction_dataset, load_and_combine_data

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Finetune a model for Korean text deobfuscation.")
    parser.add_argument("--data_dir", type=str, default="./", help="Directory containing the data files.")
    parser.add_argument("--sample_size", type=int, default=10000, help="Number of samples to use for finetuning.")
    parser.add_argument("--model_name", type=str, default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B", help="Pretrained model name from Hugging Face.")
    parser.add_argument("--output_dir", type=str, default="./hyperclova-deobfuscation-lora", help="Directory to save the finetuned model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size per device.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--test_split_size", type=float, default=0.1, help="Proportion of the dataset to use for validation.")
    return parser.parse_args()


def format_instruction(example: dict) -> dict:
    """Format the dataset into a prompt structure for the model."""
    return {
        "text": f"### 지시사항:\n{example['input']}\n\n### 응답:\n{example['output']}<|endoftext|>"
    }


def main():
    args = get_args()

    # --- 1. Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    # --- 2. Data Loading and Preprocessing ---
    combined_df = load_and_combine_data(args.data_dir)
    instruction_df = create_instruction_dataset(combined_df, args.sample_size)

    train_df, val_df = train_test_split(
        instruction_df,
        test_size=args.test_split_size,
        random_state=42,
        stratify=instruction_df['category']
    )

    print(f"\nTrain samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df)
    })
    print("\nDataset structure:")
    print(dataset)

    # --- 3. Model and Tokenizer Setup ---
    print(f"\nLoading model: {args.model_name}")

    use_quantization = device.type == 'cuda'
    model_kwargs = {"device_map": "auto"}

    if use_quantization:
        print("CUDA detected, applying 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        print("Running on MPS or CPU, quantization is disabled.")
        # MPS/CPU에서는 bfloat16을 사용하여 성능과 메모리 효율성을 높입니다.
        model_kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )

    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    # --- 4. Data Formatting ---
    formatted_dataset = dataset.map(format_instruction, remove_columns=dataset['train'].column_names)
    print("\nFormatted data example:")
    print(formatted_dataset['train'][0]['text'][:500] + "...")

    # --- 5. Model Training ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_quantization,  # fp16은 CUDA 양자화와 함께 사용할 때 가장 효과적입니다.
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset['train'],
        eval_dataset=formatted_dataset['validation'],
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nTraining finished. Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Model and tokenizer saved successfully.")


if __name__ == "__main__":
    main() 