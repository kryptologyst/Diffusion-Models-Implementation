"""Evaluation module for diffusion models."""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from clean_fid import fid
from torchmetrics.image import InceptionScore, FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import lpips

from ..models.unet import UNet
from ..diffusion import NoiseScheduler, DiffusionProcess
from ..utils import get_device


class DiffusionEvaluator:
    """Evaluator for diffusion models."""
    
    def __init__(
        self,
        model: UNet,
        scheduler: NoiseScheduler,
        device: str = "auto",
        num_samples: int = 10000,
        batch_size: int = 64,
        fid_batch_size: int = 256,
        fid_device: str = "auto",
    ):
        """Initialize evaluator.
        
        Args:
            model: Trained diffusion model.
            scheduler: Noise scheduler.
            device: Device for model inference.
            num_samples: Number of samples for evaluation.
            batch_size: Batch size for generation.
            fid_batch_size: Batch size for FID computation.
            fid_device: Device for FID computation.
        """
        self.model = model
        self.scheduler = scheduler
        self.device = get_device(device)
        self.fid_device = get_device(fid_device)
        
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.fid_batch_size = fid_batch_size
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize diffusion process
        self.diffusion = DiffusionProcess(scheduler)
        
        # Initialize metrics
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
        self.is_metric = InceptionScore()
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        
        # Initialize LPIPS
        self.lpips_model = lpips.LPIPS(net="alex").to(self.device)
    
    def generate_samples(
        self,
        num_samples: Optional[int] = None,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate.
            num_inference_steps: Number of denoising steps.
            eta: Stochasticity parameter.
            seed: Random seed.
            
        Returns:
            Generated samples.
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        if seed is not None:
            torch.manual_seed(seed)
        
        samples = []
        
        with torch.no_grad():
            for i in range(0, num_samples, self.batch_size):
                current_batch_size = min(self.batch_size, num_samples - i)
                shape = (current_batch_size, 3, 32, 32)
                
                batch_samples = self.diffusion.p_sample_loop(
                    self.model,
                    shape,
                    num_inference_steps=num_inference_steps,
                    eta=eta,
                )
                
                samples.append(batch_samples.cpu())
        
        return torch.cat(samples, dim=0)
    
    def compute_fid(
        self,
        generated_samples: torch.Tensor,
        real_samples: torch.Tensor,
    ) -> float:
        """Compute FID score.
        
        Args:
            generated_samples: Generated samples.
            real_samples: Real samples.
            
        Returns:
            FID score.
        """
        # Normalize samples to [0, 1]
        generated_samples = (generated_samples + 1) / 2
        real_samples = (real_samples + 1) / 2
        
        # Clamp to valid range
        generated_samples = torch.clamp(generated_samples, 0, 1)
        real_samples = torch.clamp(real_samples, 0, 1)
        
        # Compute FID
        self.fid_metric.update(real_samples, real=True)
        self.fid_metric.update(generated_samples, real=False)
        
        fid_score = self.fid_metric.compute().item()
        self.fid_metric.reset()
        
        return fid_score
    
    def compute_is(self, generated_samples: torch.Tensor) -> Tuple[float, float]:
        """Compute Inception Score.
        
        Args:
            generated_samples: Generated samples.
            
        Returns:
            Tuple of (mean IS, std IS).
        """
        # Normalize samples to [0, 1]
        generated_samples = (generated_samples + 1) / 2
        generated_samples = torch.clamp(generated_samples, 0, 1)
        
        # Compute IS
        self.is_metric.update(generated_samples)
        is_score = self.is_metric.compute()
        self.is_metric.reset()
        
        return is_score.item(), 0.0  # TODO: Compute std
    
    def compute_lpips_diversity(self, generated_samples: torch.Tensor) -> float:
        """Compute LPIPS diversity.
        
        Args:
            generated_samples: Generated samples.
            
        Returns:
            LPIPS diversity score.
        """
        # Normalize samples to [-1, 1] for LPIPS
        generated_samples = generated_samples.to(self.device)
        
        # Compute pairwise LPIPS distances
        distances = []
        num_samples = generated_samples.shape[0]
        
        with torch.no_grad():
            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    dist = self.lpips_model(
                        generated_samples[i:i+1],
                        generated_samples[j:j+1]
                    )
                    distances.append(dist.item())
        
        return np.mean(distances)
    
    def evaluate(
        self,
        real_dataloader: DataLoader,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        compute_fid: bool = True,
        compute_is: bool = True,
        compute_lpips: bool = True,
        save_samples: bool = True,
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Comprehensive evaluation.
        
        Args:
            real_dataloader: DataLoader with real samples.
            num_inference_steps: Number of denoising steps.
            eta: Stochasticity parameter.
            compute_fid: Whether to compute FID.
            compute_is: Whether to compute IS.
            compute_lpips: Whether to compute LPIPS diversity.
            save_samples: Whether to save generated samples.
            save_path: Path to save samples.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        print("Generating samples...")
        generated_samples = self.generate_samples(
            num_inference_steps=num_inference_steps,
            eta=eta,
        )
        
        # Collect real samples
        real_samples = []
        for batch, _ in real_dataloader:
            real_samples.append(batch)
        real_samples = torch.cat(real_samples, dim=0)
        
        # Ensure same number of samples
        min_samples = min(len(generated_samples), len(real_samples))
        generated_samples = generated_samples[:min_samples]
        real_samples = real_samples[:min_samples]
        
        print(f"Evaluating on {min_samples} samples...")
        
        metrics = {}
        
        # Compute FID
        if compute_fid:
            print("Computing FID...")
            fid_score = self.compute_fid(generated_samples, real_samples)
            metrics["fid"] = fid_score
            print(f"FID: {fid_score:.4f}")
        
        # Compute IS
        if compute_is:
            print("Computing IS...")
            is_mean, is_std = self.compute_is(generated_samples)
            metrics["is_mean"] = is_mean
            metrics["is_std"] = is_std
            print(f"IS: {is_mean:.4f} ± {is_std:.4f}")
        
        # Compute LPIPS diversity
        if compute_lpips:
            print("Computing LPIPS diversity...")
            lpips_score = self.compute_lpips_diversity(generated_samples)
            metrics["lpips_diversity"] = lpips_score
            print(f"LPIPS Diversity: {lpips_score:.4f}")
        
        # Save samples
        if save_samples:
            if save_path is None:
                save_path = "generated_samples.pt"
            
            torch.save(generated_samples, save_path)
            print(f"Generated samples saved to {save_path}")
            
            # Save sample grid
            grid_path = save_path.replace(".pt", "_grid.png")
            self.save_sample_grid(generated_samples[:64], grid_path)
            print(f"Sample grid saved to {grid_path}")
        
        return metrics
    
    def save_sample_grid(
        self,
        samples: torch.Tensor,
        save_path: str,
        nrow: int = 8,
    ) -> None:
        """Save sample grid as image.
        
        Args:
            samples: Samples to save.
            save_path: Path to save image.
            nrow: Number of images per row.
        """
        # Normalize to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # Create grid
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        
        # Convert to numpy and save
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(12, 12))
        plt.imshow(grid_np)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    def interpolate_samples(
        self,
        num_steps: int = 10,
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate interpolated samples.
        
        Args:
            num_steps: Number of interpolation steps.
            num_inference_steps: Number of denoising steps.
            eta: Stochasticity parameter.
            
        Returns:
            Interpolated samples.
        """
        # Generate two random samples
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
