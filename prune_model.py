import torch
import torch.nn.utils.prune as prune
import argparse
import os
from config import TrainingConfig, HardwareConfig
from src.lora_model import LoRAModel

def apply_structured_pruning(model, amount=0.5):
    """Apply structured pruning to all Linear layers in the model."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, 'weight')
    return model

def main():
    parser = argparse.ArgumentParser(description="Prune a trained LLaMA model.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--prune-amount', type=float, default=0.5, help='Fraction of weights to prune (default: 0.5)')
    parser.add_argument('--output', type=str, default='checkpoints/pruned_model.pt', help='Path to save the pruned model')
    args = parser.parse_args()

    config = TrainingConfig()
    hardware_config = HardwareConfig()
    model_handler = LoRAModel(config)
    model, _ = model_handler.load_checkpoint(args.checkpoint, local_rank=0)

    print(f"Applying structured pruning with amount={args.prune_amount}...")
    pruned_model = apply_structured_pruning(model, amount=args.prune_amount)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'model_state_dict': pruned_model.state_dict()}, args.output)
    print(f"Pruned model saved to {args.output}")

if __name__ == "__main__":
    main() 