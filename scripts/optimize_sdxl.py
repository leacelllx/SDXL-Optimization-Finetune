import json
import argparse
import torch
from diffusers import StableDiffusionPipeline
from torch.nn.utils import prune

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

# Function to load generation queue from JSON file
def load_generation_queue(file_path='generation_queue.json'):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data["generation_queue"]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading generation queue: {e}")
        return []

# Function to generate images based on prompts and seeds
def generate_images(queue):
    pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        use_safetensors=True,
        torch_dtype=torch.float16,
        variant='fp16'
    ).to('cuda')
    
    generator = torch.Generator(device='cuda')
    for i, generation in enumerate(queue, start=1):
        generator.manual_seed(generation['seed'])
        image = pipe(prompt=generation['prompt'], generator=generator).images[0]
        image.save(f'image_{i}.png')
        print(f"Generated and saved image_{i}.png")

# Optimization functions
def apply_precision_reduction(model, precision):
    if precision == "FP16":
        model.half()
    return model

def apply_efficient_attention(model, method):
    if method == "xformers":
        try:
            import xformers
            from xformers.ops import memory_efficient_attention
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.forward = memory_efficient_attention
        except ImportError:
            print("xFormers is not installed. Please install it to use this feature.")
    return model

def apply_layer_pruning(model, prune_percentage):
    parameters_to_prune = [(module, 'weight') for name, module in model.named_modules() 
                           if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_percentage)
    return model

# Main optimization function
def optimize_model(config, apply_all, precision, attention, pruning):
    model = SDXLModel()  # Assuming SDXLModel is imported and defined elsewhere
    model.load_state_dict(torch.load(config['model_path']))

    if apply_all or precision:
        if config['optimization_techniques']['precision_reduction']['enabled']:
            model = apply_precision_reduction(model, config['optimization_techniques']['precision_reduction']['precision'])

    if apply_all or attention:
        if config['optimization_techniques']['efficient_attention']['enabled']:
            model = apply_efficient_attention(model, config['optimization_techniques']['efficient_attention']['method'])

    if apply_all or pruning:
        if config['optimization_techniques']['layer_pruning']['enabled']:
            model = apply_layer_pruning(model, config['optimization_techniques']['layer_pruning']['prune_percentage'])

    torch.save(model.state_dict(), config['output_path'])
    print(f"Optimized model saved to {config['output_path']}")

# Main function to manage generation and optimization
def main():
    parser = argparse.ArgumentParser(description="Optimize and generate images with SDXL")
    parser.add_argument('--config', type=str, required=True, help="Path to the optimization config file")
    parser.add_argument('--all', action='store_true', help="Apply all optimizations")
    parser.add_argument('--precision', action='store_true', help="Apply precision reduction")
    parser.add_argument('--attention', action='store_true', help="Apply efficient attention mechanism")
    parser.add_argument('--pruning', action='store_true', help="Apply layer pruning")
    parser.add_argument('--generate', action='store_true', help="Generate images from the queue in generation_queue.json")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.generate:
        queue = load_generation_queue()
        if queue:
            generate_images(queue)
        else:
            print("The generation queue is empty or could not be loaded.")
    else:
        optimize_model(config, args.all, args.precision, args.attention, args.pruning)

if __name__ == "__main__":
    main()
