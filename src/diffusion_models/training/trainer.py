"""Training module for diffusion models."""

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from omegaconf import DictConfig

from ..models.unet import UNet
from ..diffusion import NoiseScheduler, DiffusionProcess
from ..utils import get_device, count_parameters


class DiffusionTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for diffusion models."""
    
    def __init__(
        self,
        model_config: DictConfig,
        diffusion_config: DictConfig,
        training_config: DictConfig,
        evaluation_config: DictConfig,
        **kwargs: Any,
    ):
        """Initialize diffusion trainer.
        
        Args:
            model_config: Model configuration.
            diffusion_config: Diffusion process configuration.
            training_config: Training configuration.
            evaluation_config: Evaluation configuration.
            **kwargs: Additional arguments.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store configurations
        self.model_config = model_config
        self.diffusion_config = diffusion_config
        self.training_config = training_config
        self.evaluation_config = evaluation_config
        
        # Initialize model
        self.model = UNet(**model_config)
        
        # Initialize diffusion process
        self.scheduler = NoiseScheduler(**diffusion_config)
        self.diffusion = DiffusionProcess(self.scheduler)
        
        # Training parameters
        self.learning_rate = training_config.learning_rate
        self.weight_decay = training_config.weight_decay
        self.beta1 = training_config.beta1
        self.beta2 = training_config.beta2
        self.eps = training_config.eps
        
        # Loss function
        self.loss_type = diffusion_config.loss_type
        if self.loss_type == "mse":
            self.criterion = F.mse_loss
        elif self.loss_type == "l1":
            self.criterion = F.l1_loss
        elif self.loss_type == "huber":
            self.criterion = F.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Logging
        self.use_wandb = kwargs.get("use_wandb", False)
        self.log_images_every_n_steps = kwargs.get("log_images_every_n_steps", 1000)
        
        # Validation
        self.val_samples = None
        self.val_noise = None
        
        # Print model info
        num_params = count_parameters(self.model)
        print(f"Model initialized with {num_params:,} parameters")
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            timesteps: Timestep tensor.
            
        Returns:
            Model output.
        """
        return self.model(x, timesteps)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Training loss.
        """
        images, _ = batch
        
        # Sample random timesteps
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,), device=self.device
        )
        
        # Sample noise
        noise = torch.randn_like(images)
        
        # Add noise to images
        noisy_images, target_noise = self.diffusion.q_sample(images, timesteps, noise)
        
        # Predict noise
        predicted_noise = self.forward(noisy_images, timesteps)
        
        # Compute loss
        if self.diffusion_config.prediction_type == "epsilon":
            loss = self.criterion(predicted_noise, target_noise)
        elif self.diffusion_config.prediction_type == "v_prediction":
            velocity = self.scheduler.get_velocity(images, noise, timesteps)
            loss = self.criterion(predicted_noise, velocity)
        else:
            loss = self.criterion(predicted_noise, images)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        
        # Log images occasionally
        if batch_idx % self.log_images_every_n_steps == 0:
            self._log_images(images, noisy_images, predicted_noise, "train")
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Validation loss.
        """
        images, _ = batch
        
        # Sample random timesteps
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,), device=self.device
        )
        
        # Sample noise
        noise = torch.randn_like(images)
        
        # Add noise to images
        noisy_images, target_noise = self.diffusion.q_sample(images, timesteps, noise)
        
        # Predict noise
        predicted_noise = self.forward(noisy_images, timesteps)
        
        # Compute loss
        if self.diffusion_config.prediction_type == "epsilon":
            loss = self.criterion(predicted_noise, target_noise)
        elif self.diffusion_config.prediction_type == "v_prediction":
            velocity = self.scheduler.get_velocity(images, noise, timesteps)
            loss = self.criterion(predicted_noise, velocity)
        else:
            loss = self.criterion(predicted_noise, images)
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store samples for logging
        if batch_idx == 0:
            self.val_samples = images[:8].detach()
            self.val_noise = noise[:8].detach()
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if self.val_samples is not None:
            self._log_images(self.val_samples, None, None, "val")
    
    def _log_images(
        self,
        original: torch.Tensor,
        noisy: Optional[torch.Tensor],
        predicted: Optional[torch.Tensor],
        stage: str,
    ) -> None:
        """Log images to wandb.
        
        Args:
            original: Original images.
            noisy: Noisy images.
            predicted: Predicted noise/images.
            stage: Training stage.
        """
        if not self.use_wandb:
            return
        
        # Denormalize images
        def denormalize(x):
            return (x + 1) / 2
        
        images = []
        
        # Original images
        orig_imgs = denormalize(original[:8])
        images.extend([wandb.Image(img) for img in orig_imgs])
        
        # Noisy images (if provided)
        if noisy is not None:
            noisy_imgs = denormalize(noisy[:8])
            images.extend([wandb.Image(img) for img in noisy_imgs])
        
        # Predicted images (if provided)
        if predicted is not None:
            if self.diffusion_config.prediction_type == "epsilon":
                # Reconstruct from predicted noise
                timesteps = torch.zeros(8, device=self.device, dtype=torch.long)
                alpha_t = self.scheduler.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
                sqrt_one_minus_alpha_t = self.scheduler.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
                reconstructed = (noisy[:8] - sqrt_one_minus_alpha_t * predicted[:8]) / torch.sqrt(alpha_t)
            else:
                reconstructed = predicted[:8]
            
            recon_imgs = denormalize(reconstructed)
            images.extend([wandb.Image(img) for img in recon_imgs])
        
        wandb.log({f"{stage}/images": images}, step=self.global_step)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers.
        
        Returns:
            Optimizer and scheduler configuration.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.training_config.max_epochs,
            eta_min=self.learning_rate * 0.01,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def on_train_start(self) -> None:
        """Called at the start of training."""
        if self.use_wandb and self.logger is not None:
            wandb.watch(self.model, log="all", log_freq=100)
    
    def generate_samples(
        self,
        num_samples: int = 16,
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate.
            num_inference_steps: Number of denoising steps.
            eta: Stochasticity parameter.
            
        Returns:
            Generated samples.
        """
        self.eval()
        with torch.no_grad():
            shape = (num_samples, self.model_config.in_channels, 32, 32)
            samples = self.diffusion.p_sample_loop(
                self.model,
                shape,
                num_inference_steps=num_inference_steps,
                eta=eta,
            )
        return samples
