import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from src.distributed import DistributedSetup
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from contextlib import contextmanager

@contextmanager
def fp8_context():
    """Context manager for FP8 training"""
    try:
        with te.fp8_autocast(enabled=True):
            yield
    except Exception as e:
        print(f"FP8 context error: {e}")
        yield

class LoRAModel:
    def __init__(self, config):
        self.config = config
        self.hw_config = config.hardware

        # Set appropriate dtype based on precision and hardware
        if self.hw_config.precision == "fp8" and self.hw_config.device_type == "cuda":
            self.dtype = torch.float32  # TE handles FP8 casting
            self.use_fp8 = True
        elif self.hw_config.precision == "bf16" and self.hw_config.device_type != "cpu":
            self.dtype = torch.bfloat16
            self.use_fp8 = False
        elif self.hw_config.precision == "fp16" and self.hw_config.device_type != "cpu":
            self.dtype = torch.float16
            self.use_fp8 = False
        else:
            self.dtype = torch.float32
            self.use_fp8 = False

    def setup_model(self, local_rank):
        # Ensure all processes use the same initialization
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Set up tokenizer with proper padding
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token to be the EOS token

        # Set deterministic initialization
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        # Initialize model with FP8 support if available
        if self.use_fp8:
            with te.fp8_autocast(enabled=True):
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=self.dtype,
                    device_map=None,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.dtype,
                device_map=None,
            )

        # Make sure model knows about padding token
        model.config.pad_token_id = tokenizer.pad_token_id

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        lora_config = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config.lora.target_modules,
            init_lora_weights=True,
        )

        if self.use_fp8:
            with te.fp8_autocast(enabled=True):
                model = get_peft_model(model, lora_config)
        else:
            model = get_peft_model(model, lora_config)

        # Synchronize after LoRA application
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return model, tokenizer

    def load_checkpoint(self, checkpoint_path, local_rank):
        """Load checkpoint with both model and optimizer states"""
        model, tokenizer = self.setup_model(local_rank)

        # Load checkpoint to CPU first to avoid GPU memory issues
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if self.use_fp8:
            with te.fp8_autocast(enabled=True):
                # Load model state dict
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                elif "lora_state_dict" in checkpoint:
                    print("Loading LoRA weights!")
                    model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
                else:
                    model.load_state_dict(
                        checkpoint, strict=False
                    )  # If checkpoint is just the model state dict
                # Move model to appropriate device
                model = model.to(DistributedSetup.get_device(self.hw_config, local_rank))
        else:
            # Load model state dict
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            elif "lora_state_dict" in checkpoint:
                print("Loading LoRA weights!")
                model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
            else:
                model.load_state_dict(
                    checkpoint, strict=False
                )  # If checkpoint is just the model state dict
            # Move model to appropriate device
            model = model.to(DistributedSetup.get_device(self.hw_config, local_rank))

        return model, tokenizer
