"""Diffusion process implementation."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseScheduler:
    """Noise scheduler for diffusion process.
    
    Implements various noise schedules including linear and cosine schedules.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
    ):
        """Initialize noise scheduler.
        
        Args:
            num_timesteps: Number of diffusion timesteps.
            beta_start: Starting value of beta.
            beta_end: Ending value of beta.
            beta_schedule: Type of beta schedule ("linear" or "cosine").
            prediction_type: Type of prediction ("epsilon", "v_prediction", or "sample").
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        
        # Compute betas
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Compute coefficients for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Compute coefficients for posterior sampling
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule as proposed in Improved DDPM.
        
        Args:
            timesteps: Number of timesteps.
            s: Small offset to prevent beta from being too small.
            
        Returns:
            Beta values.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=0.001, max=0.999)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to original samples.
        
        Args:
            original_samples: Original clean samples.
            noise: Random noise tensor.
            timesteps: Diffusion timesteps.
            
        Returns:
            Noisy samples.
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        
        noisy_samples = sqrt_alphas_cumprod_t * original_samples + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_samples
    
    def get_velocity(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Get velocity target for v-prediction.
        
        Args:
            original_samples: Original clean samples.
            noise: Random noise tensor.
            timesteps: Diffusion timesteps.
            
        Returns:
            Velocity targets.
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        
        velocity = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * original_samples
        return velocity
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Single denoising step.
        
        Args:
            model_output: Model prediction.
            timestep: Current timestep.
            sample: Current noisy sample.
            eta: Stochasticity parameter for DDIM.
            
        Returns:
            Denoised sample.
        """
        prev_timestep = timestep - self.num_timesteps // self.num_timesteps
        
        # Compute coefficients
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Compute predicted original sample
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        else:
            pred_original_sample = model_output
        
        # Compute variance
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** 0.5
        
        # Compute previous sample
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + beta_prod_t_prev ** 0.5 * model_output
        
        if eta > 0:
            noise = torch.randn_like(model_output)
            prev_sample = prev_sample + std_dev_t * noise
        
        return prev_sample


class DiffusionProcess:
    """Main diffusion process class."""
    
    def __init__(self, scheduler: NoiseScheduler):
        """Initialize diffusion process.
        
        Args:
            scheduler: Noise scheduler.
        """
        self.scheduler = scheduler
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process.
        
        Args:
            x_start: Starting samples.
            t: Timesteps.
            noise: Optional noise tensor.
            
        Returns:
            Tuple of (noisy samples, noise).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        noisy_samples = self.scheduler.add_noise(x_start, noise, t)
        return noisy_samples, noise
    
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Single denoising step.
        
        Args:
            model: Diffusion model.
            x: Current noisy sample.
            t: Current timestep.
            clip_denoised: Whether to clip denoised values.
            eta: Stochasticity parameter.
            
        Returns:
            Denoised sample.
        """
        with torch.no_grad():
            model_output = model(x, t)
            prev_sample = self.scheduler.step(model_output, t.item(), x, eta)
            
            if clip_denoised:
                prev_sample = torch.clamp(prev_sample, -1.0, 1.0)
            
            return prev_sample
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Generate samples using reverse diffusion process.
        
        Args:
            model: Diffusion model.
            shape: Shape of samples to generate.
            num_inference_steps: Number of denoising steps.
            eta: Stochasticity parameter.
            clip_denoised: Whether to clip denoised values.
            
        Returns:
            Generated samples.
        """
        device = next(model.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Create timestep schedule
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device
        )
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, clip_denoised, eta)
        
        return x
