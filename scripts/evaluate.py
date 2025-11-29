#!/usr/bin/env python3
"""Evaluation script for diffusion models."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import DictConfig, OmegaConf

from src.diffusion_models.evaluation.evaluator import DiffusionEvaluator
from src.diffusion_models.data.cifar10 import CIFAR10DataModule
from src.diffusion_models.models.unet import UNet
from src.diffusion_models.diffusion import NoiseScheduler
from src.diffusion_models.utils import get_device, load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output", type=str, default="evaluation_results.txt", help="Output file")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples for evaluation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--eta", type=float, default=0.0, help="Stochasticity parameter")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--compute_fid", action="store_true", help="Compute FID")
    parser.add_argument("--compute_is", action="store_true", help="Compute IS")
    parser.add_argument("--compute_lpips", action="store_true", help="Compute LPIPS diversity")
    parser.add_argument("--save_samples", action="store_true", help="Save generated samples")
    
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    device: str = "auto",
) -> tuple[UNet, NoiseScheduler]:
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
    config = load_config(config_path)
    
    # Initialize model
    model = UNet(**config.model)
    
    # Initialize scheduler
    scheduler = NoiseScheduler(**config.diffusion)
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, scheduler


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load config
    if args.config is None:
        # Try to find config in checkpoint
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = checkpoint.get("config")
        if config is None:
            raise ValueError("No config found in checkpoint and no config_path provided")
    else:
        config = load_config(args.config)
    
    # Setup device
    device = get_device(args.device)
    
    # Load model
    print("Loading model...")
    model, scheduler = load_model_from_checkpoint(
        args.checkpoint,
        args.config or "configs/config.yaml",
        args.device,
    )
    
    # Setup data module
    data_module = CIFAR10DataModule(**config.data)
    data_module.prepare_data()
    data_module.setup("test")
    
    # Setup evaluator
    evaluator = DiffusionEvaluator(
        model=model,
        scheduler=scheduler,
        device=args.device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
    
    # Run evaluation
    print("Running evaluation...")
    metrics = evaluator.evaluate(
        real_dataloader=data_module.test_dataloader(),
        num_inference_steps=args.num_steps,
        eta=args.eta,
        compute_fid=args.compute_fid,
        compute_is=args.compute_is,
        compute_lpips=args.compute_lpips,
        save_samples=args.save_samples,
        save_path="generated_samples.pt" if args.save_samples else None,
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    with open(args.output, "w") as f:
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
