import torch
import torch.nn.functional as F
import argparse
import os
from config import TrainingConfig, HardwareConfig
from src.lora_model import LoRAModel
from src.dataset import load_dataset_for_training
from transformers import AutoTokenizer

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """KL divergence between softened student and teacher logits."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

def main():
    parser = argparse.ArgumentParser(description="Distill a LLaMA model (student) from a teacher.")
    parser.add_argument('--teacher-checkpoint', type=str, required=True, help='Path to the teacher model checkpoint')
    parser.add_argument('--student-model-name', type=str, default='meta-llama/Llama-3.1-3B', help='Student model name (default: LLaMA 3B)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of distillation epochs')
    parser.add_argument('--output', type=str, default='checkpoints/distilled_student.pt', help='Path to save the distilled student model')
    args = parser.parse_args()

    # Load config and dataset
    config = TrainingConfig()
    hardware_config = HardwareConfig()
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_name)
    train_dataset = load_dataset_for_training(
        dataset_name=config.speed_dataset,
        tokenizer=tokenizer,
        prompt_config=config.prompt,
        split="all",
        max_length=config.max_length,
    )

    # Load teacher
    teacher_config = TrainingConfig()
    teacher_config.model_name = config.model_name  # Use main config's teacher model
    teacher_handler = LoRAModel(teacher_config)
    teacher, _ = teacher_handler.load_checkpoint(args.teacher_checkpoint, local_rank=0)
    teacher.eval()

    # Load student
    student_config = TrainingConfig()
    student_config.model_name = args.student_model_name
    student_handler = LoRAModel(student_config)
    student, _ = student_handler.setup_model(local_rank=0)
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for batch in train_dataset:
            inputs = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
            with torch.no_grad():
                teacher_logits = teacher(**inputs).logits
            student_logits = student(**inputs).logits
            loss = distillation_loss(student_logits, teacher_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'model_state_dict': student.state_dict()}, args.output)
    print(f"Distilled student model saved to {args.output}")

if __name__ == "__main__":
    main() 