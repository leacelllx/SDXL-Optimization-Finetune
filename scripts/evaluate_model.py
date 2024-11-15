import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset, load_metric

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

# Load dataset and prepare for evaluation
def load_eval_dataset(dataset_name, split="validation"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset.map(lambda example: tokenizer(example['text'], truncation=True, padding="max_length"), batched=True)

# Evaluate model
def evaluate_model(config):
    model = AutoModelForCausalLM.from_pretrained(config['model_path'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])

    eval_dataset = load_eval_dataset(config['dataset_name'], split=config['dataset_split'])

    # Initialize Trainer for evaluation
    args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_eval_batch_size=config['batch_size']
    )

    metric = load_metric(config['metric_name'])  # e.g., "accuracy" or "perplexity"
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(dim=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator
    )

    results = trainer.evaluate()
    print("Evaluation results:", results)
    return results

# Main entry point
if __name__ == "__main__":
    config = load_config('evaluate_config.json')
    evaluate_model(config)
