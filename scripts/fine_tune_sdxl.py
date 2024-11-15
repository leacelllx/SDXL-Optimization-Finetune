import json
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

# Prepare dataset
def load_dataset_for_finetuning(dataset_name, split="train"):
    dataset = load_dataset(dataset_name, split=split)
    def tokenize_function(example):
        return tokenizer(example['text'], padding='max_length', truncation=True)
    return dataset.map(tokenize_function, batched=True)

# Fine-tune model
def fine_tune_model(config):
    model = AutoModelForCausalLM.from_pretrained(config['base_model'])
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])

    dataset = load_dataset_for_finetuning(config['dataset_name'], split=config['dataset_split'])
    
    args = TrainingArguments(
        output_dir=config['output_dir'],
        evaluation_strategy="epoch",
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
        weight_decay=config['weight_decay'],
        fp16=config['fp16']
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(config['output_dir'])
    print(f"Fine-tuned model saved to {config['output_dir']}")

# Main entry point
if __name__ == "__main__":
    config = load_config('fine_tuning_config.json')
    fine_tune_model(config)
