"""Sampling utilities for diffusion models."""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from ..models.unet import UNet
from ..diffusion import NoiseScheduler, DiffusionProcess
from ..utils import get_device, set_seed, load_config


class DiffusionSampler:
    """Sampler for diffusion models."""
    
    def __init__(
        self,
        model: UNet,
        scheduler: NoiseScheduler,
        device: str = "auto",
    ):
        """Initialize sampler.
        
        Args:
            model: Trained diffusion model.
            scheduler: Noise scheduler.
            device: Device for sampling.
        """
        self.model = model
        self.scheduler = scheduler
        self.device = get_device(device)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize diffusion process
        self.diffusion = DiffusionProcess(scheduler)
    
    def sample(
        self,
        num_samples: int = 16,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples.
        
        Args:
            num_samples: Number of samples to generate.
            num_inference_steps: Number of denoising steps.
            eta: Stochasticity parameter.
            guidance_scale: Guidance scale for classifier-free guidance.
            seed: Random seed.
            
        Returns:
            Generated samples.
        """
        if seed is not None:
            set_seed(seed)
        
        shape = (num_samples, 3, 32, 32)
        
        with torch.no_grad():
            samples = self.diffusion.p_sample_loop(
                self.model,
                shape,
                num_inference_steps=num_inference_steps,
                eta=eta,
            )
        
        return samples
    
    def sample_with_progress(
        self,
        num_samples: int = 16,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate samples with progress bar.
        
        Args:
            num_samples: Number of samples to generate.
            num_inference_steps: Number of denoising steps.
            eta: Stochasticity parameter.
            seed: Random seed.
            show_progress: Whether to show progress.
            
        Returns:
            Generated samples.
        """
        if seed is not None:
            set_seed(seed)
        
        device = self.device
        b = num_samples
        
        # Start from pure noise
        x = torch.randn((b, 3, 32, 32), device=device)
        
        # Create timestep schedule
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device
        )
        
        # Denoising loop with progress
        if show_progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling")
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = torch.full((b,), t, device=device, dtype=torch.long)
                x = self.diffusion.p_sample(self.model, x, t_batch, eta=eta)
        
        return x
    
    def interpolate(
        self,
        num_steps: int = 10,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate interpolated samples.
        
        Args:
            num_steps: Number of interpolation steps.
            num_inference_steps: Number of denoising steps.
            eta: Stochasticity parameter.
            seed: Random seed.
            
        Returns:
            Interpolated samples.
        """
        if seed is not None:
            set_seed(seed)
        
        # Generate two random noise vectors
        z1 = torch.randn(1, 3, 32, 32, device=self.device)
        z2 = torch.randn(1, 3, 32, 32, device=self.device)
        
        # Create interpolation weights
        weights = torch.linspace(0, 1, num_steps, device=self.device)
        
        interpolated_samples = []
        
        with torch.no_grad():
            for w in weights:
                # Interpolate in noise space
                z_interp = (1 - w) * z1 + w * z2
                
                # Generate sample
                sample = self.diffusion.p_sample_loop(
                    self.model,
                    z_interp.shape,
                    num_inference_steps=num_inference_steps,
                    eta=eta,
                )
                
                interpolated_samples.append(sample.cpu())
        
        return torch.cat(interpolated_samples, dim=0)
    
    def save_samples(
        self,
        samples: torch.Tensor,
        save_path: str,
        nrow: int = 8,
        normalize: bool = True,
    ) -> None:
        """Save samples as image grid.
        
        Args:
            samples: Samples to save.
            save_path: Path to save image.
            nrow: Number of images per row.
            normalize: Whether to normalize images.
        """
        # Create grid
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=normalize)
        
        # Save image
        save_image(grid, save_path)
    
    def save_interpolation(
        self,
        samples: torch.Tensor,
        save_path: str,
        nrow: Optional[int] = None,
    ) -> None:
        """Save interpolation as image.
        
        Args:
            samples: Interpolated samples.
            save_path: Path to save image.
            nrow: Number of images per row (default: all in one row).
        """
        if nrow is None:
            nrow = len(samples)
        
        # Create grid
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=True)
        
        # Save image
        save_image(grid, save_path)


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "auto",
) -> Tuple[UNet, NoiseScheduler]:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to config file.
        device: Device to load model on.
        
    Returns:
        Tuple of (model, scheduler).
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load config
    if config_path is None:
        config = checkpoint.get("config")
        if config is None:
            raise ValueError("No config found in checkpoint and no config_path provided")
    else:
        config = load_config(config_path)
    
    # Initialize model
    model = UNet(**config.model)
    
    # Initialize scheduler
    scheduler = NoiseScheduler(**config.diffusion)
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, scheduler


def main():
    """Main sampling script."""
    parser = argparse.ArgumentParser(description="Sample from diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output", type=str, default="samples.png", help="Output path")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--eta", type=float, default=0.0, help="Stochasticity parameter")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--interpolate", action="store_true", help="Generate interpolation")
    parser.add_argument("--interp_steps", type=int, default=10, help="Number of interpolation steps")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, scheduler = load_model_from_checkpoint(
        args.checkpoint,
        args.config,
        args.device,
    )
    
    # Create sampler
    sampler = DiffusionSampler(model, scheduler, args.device)
    
    # Generate samples
    if args.interpolate:
        print("Generating interpolation...")
        samples = sampler.interpolate(
            num_steps=args.interp_steps,
            num_inference_steps=args.num_steps,
            eta=args.eta,
            seed=args.seed,
        )
        sampler.save_interpolation(samples, args.output)
    else:
        print("Generating samples...")
        samples = sampler.sample(
            num_samples=args.num_samples,
            num_inference_steps=args.num_steps,
            eta=args.eta,
            seed=args.seed,
        )
        sampler.save_samples(samples, args.output)
    
    print(f"Samples saved to {args.output}")


if __name__ == "__main__":
    main()
