"""Example usage of diffusion models."""

import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from src.diffusion_models.models.unet import UNet
from src.diffusion_models.diffusion import NoiseScheduler, DiffusionProcess
from src.diffusion_models.sampling.sampler import DiffusionSampler
from src.diffusion_models.utils import set_seed, get_device


def main():
    """Example usage of diffusion models."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Load configuration
    config = OmegaConf.load("configs/config.yaml")
    
    # Initialize model
    model = UNet(**config.model)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize scheduler
    scheduler = NoiseScheduler(**config.diffusion)
    
    # Initialize diffusion process
    diffusion = DiffusionProcess(scheduler)
    
    # Create sampler
    sampler = DiffusionSampler(model, scheduler, "auto")
    
    # Generate samples
    print("Generating samples...")
    samples = sampler.sample(
        num_samples=16,
        num_inference_steps=50,
        eta=0.0,
        seed=42,
    )
    
    # Convert to numpy and denormalize
    samples_np = samples.cpu().numpy()
    samples_np = (samples_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    samples_np = np.clip(samples_np, 0, 1)
    
    # Display samples
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(samples_np):
            ax.imshow(samples_np[i].transpose(1, 2, 0))
            ax.axis("off")
        else:
            ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("example_samples.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    print("Example completed! Check 'example_samples.png' for generated samples.")


if __name__ == "__main__":
    main()
