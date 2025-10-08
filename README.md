# ISC25 Student Cluster Competition: LLaMA Fine-Tuning Task

## Performance Optimization Techniques

To maximize training throughput on 8× H100 GPUs, we employed the following performance-oriented strategies:

- **FlashAttention 3**: Leveraged H100-specific kernel optimizations for faster attention computation. This significantly reduced time per step while maintaining numerical stability.
  
- **Transformer Engine**: Enabled NVIDIA’s optimized Transformer Engine with FP8 (as opposed to the currently supported full precision or fp/bf16) support. This allowed layers to run with reduced memory usage and improved compute efficiency via autocasting and fused operations. See below:

- **Precision**: Adopted FP8 precision for training to reduce memory footprint and accelerate computation. FP8 was chosen over BF16 due to its superior performance on H100s, especially when combined with the Transformer Engine. It also allow for the below to be in the way it is:

- **Effective Batch Size**: Set to 256 through a configuration of `per_device_batch_size=16`, `num_gpus=8`, and `gradient_accumulation_steps=2`. This allowed for stable convergence while remaining within memory constraints.

## Accuracy Optimization Techniques

To improve downstream task performance, we adopted:

- **DoRA (Delta-Orthogonal Rank Adaptation)**: A recent fine-tuning technique that improves upon LoRA and qLoRA by maintaining full-rank parameter contributions while introducing minimal training overhead. We observed higher validation accuracy with DoRA over qLoRA in our early experiments.

- **Epochs and Max Steps**: We trained for 5 epochs with a capped `max_steps=69` to ensure coverage of the dataset within our time budget. This setting was selected based on convergence behavior and cluster runtime availability. We take effective batch size 256 and estimated sequence length 512 => tokens per step = 256 × 512 = 131072 tokens => Steps per epoch = 9,000,000 / 131072 ≈ 69 steps per epoch => 5 epochs	~685 steps - we see this as a reasonable amount to improve accuracy withtout overfitting.

## Discussion of Trade-Offs and Decisions

We evaluated multiple strategies and made the following informed decisions:

- **DoRA over LoRA/qLoRA**: While qLoRA is more memory efficient, DoRA offered higher accuracy with moderate compute overhead. Given our  8 × H100s, we prioritized final model quality over minimal resource use.

- **FlashAttention 3 and Transformer Engine**: Both significantly increased throughput, but required careful integration and validation due to evolving support and dependency issues. We chose to accept this complexity for the performance gain. (Note we appreciate the ease of transformer engine)

- **Precision: FP8 vs. BF16**: FP8 enabled larger batch sizes and faster execution. Although convergence in FP8 can be less stable in some settings, pairing it with Transformer Engine ensured reliable training.

- **Batch Size & Training Schedule**: Our batch size and gradient accumulation steps were chosen to fit within the 96GB H100 VRAM while maximizing throughput. The step count and epoch configuration provided a good balance between convergence and runtime feasibility within the competition limits.

## Note to readers post compeition

Note all files are in the states they were at final submission. I aim to put them to the correct state in the future.
