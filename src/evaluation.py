import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .dataset import get_data_collator


logger = logging.getLogger(__name__)


class CausalLMEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.base_model = model.module if hasattr(model, "module") else model
        self.tokenizer = tokenizer
        self.data_collator = get_data_collator(tokenizer)

        # Get device from model's parameters instead of assuming
        self.device = next(self.base_model.parameters()).device

        self.model.eval()  # Set model to evaluation mode

    # Please do not modify the evaluate function
    def evaluate(self, dataset, batch_size=8):
        """
        Memory-optimized evaluation
        """
        # Configure DataLoader for memory efficiency
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Reduce memory usage by not using multiple workers
            pin_memory=False,  # Disable pin_memory to reduce memory usage
            drop_last=False,
            collate_fn=self.data_collator,
        )

        total = 0
        correct = 0

        # Use tqdm only on rank 0
        is_main_process = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        dataloader = (
            tqdm(dataloader, desc="Evaluating") if is_main_process else dataloader
        )

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device efficiently
                batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }

                # Generate predictions in smaller chunks if needed
                try:
                    outputs = self.base_model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # If OOM, try with half the batch
                        torch.cuda.empty_cache()
                        half_batch_size = batch["input_ids"].size(0) // 2

                        # Process first half
                        outputs1 = self.base_model.generate(
                            input_ids=batch["input_ids"][:half_batch_size],
                            attention_mask=batch["attention_mask"][:half_batch_size],
                            max_new_tokens=1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )

                        # Process second half
                        outputs2 = self.base_model.generate(
                            input_ids=batch["input_ids"][half_batch_size:],
                            attention_mask=batch["attention_mask"][half_batch_size:],
                            max_new_tokens=1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )

                        outputs = torch.cat([outputs1, outputs2], dim=0)
                    else:
                        raise e

                # Get predictions and update metrics
                predictions = outputs[:, -1].to(self.device)
                labels = batch["labels"][:, -1].to(self.device)

                # Update metrics
                valid_mask = labels != -100
                predictions = predictions[valid_mask]
                labels = labels[valid_mask]

                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                # Clear cache after each batch
                torch.cuda.empty_cache()

        accuracy = correct / total if total > 0 else 0

        # Gather metrics across all processes
        if torch.distributed.is_initialized():
            metrics = torch.tensor([correct, total], device=self.device)
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            correct, total = metrics.tolist()

        # Calculate final metrics
        accuracy = correct / total if total > 0 else 0
        if is_main_process:
            print("\nFinal Results:")
            print(f"Total correct predictions: {correct}")
            print(f"Total samples: {total}")
            print(f"Accuracy: {accuracy:.4f}")

        return {
            "accuracy": accuracy,
            "correct_predictions": correct,
            "total_samples": total,
        }
