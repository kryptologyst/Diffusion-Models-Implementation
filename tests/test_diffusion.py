"""Unit tests for diffusion models."""

import pytest
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.diffusion_models.models.unet import UNet
from src.diffusion_models.diffusion import NoiseScheduler, DiffusionProcess
from src.diffusion_models.utils import set_seed, get_device, count_parameters


class TestUNet:
    """Test UNet model."""
    
    def test_unet_forward(self):
        """Test UNet forward pass."""
        model = UNet(
            in_channels=3,
            out_channels=3,
            model_channels=64,
            attention_resolutions=[16],
            num_res_blocks=1,
            channel_mult=[1, 2],
            num_heads=2,
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        timesteps = torch.randint(0, 1000, (2,))
        
        output = model(x, timesteps)
        
        assert output.shape == x.shape
        assert output.dtype == x.dtype
    
    def test_unet_parameters(self):
        """Test UNet parameter count."""
        model = UNet(
            in_channels=3,
            out_channels=3,
            model_channels=64,
            attention_resolutions=[16],
            num_res_blocks=1,
            channel_mult=[1, 2],
            num_heads=2,
        )
        
        num_params = count_parameters(model)
        assert num_params > 0
        assert isinstance(num_params, int)


class TestNoiseScheduler:
    """Test noise scheduler."""
    
    def test_noise_scheduler_init(self):
        """Test noise scheduler initialization."""
        scheduler = NoiseScheduler(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )
        
        assert scheduler.num_timesteps == 100
        assert len(scheduler.betas) == 100
        assert len(scheduler.alphas) == 100
        assert len(scheduler.alphas_cumprod) == 100
    
    def test_add_noise(self):
        """Test noise addition."""
        scheduler = NoiseScheduler(num_timesteps=100)
        
        original = torch.randn(2, 3, 32, 32)
        noise = torch.randn(2, 3, 32, 32)
        timesteps = torch.randint(0, 100, (2,))
        
        noisy = scheduler.add_noise(original, noise, timesteps)
        
        assert noisy.shape == original.shape
        assert noisy.dtype == original.dtype
    
    def test_cosine_schedule(self):
        """Test cosine beta schedule."""
        scheduler = NoiseScheduler(
            num_timesteps=100,
            beta_schedule="cosine",
        )
        
        assert len(scheduler.betas) == 100
        assert torch.all(scheduler.betas >= 0)
        assert torch.all(scheduler.betas <= 1)


class TestDiffusionProcess:
    """Test diffusion process."""
    
    def test_q_sample(self):
        """Test forward diffusion process."""
        scheduler = NoiseScheduler(num_timesteps=100)
        diffusion = DiffusionProcess(scheduler)
        
        x_start = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 100, (2,))
        
        noisy, noise = diffusion.q_sample(x_start, t)
        
        assert noisy.shape == x_start.shape
        assert noise.shape == x_start.shape
        assert noisy.dtype == x_start.dtype
    
    def test_p_sample_loop(self):
        """Test reverse diffusion process."""
        scheduler = NoiseScheduler(num_timesteps=100)
        diffusion = DiffusionProcess(scheduler)
        
        model = UNet(
            in_channels=3,
            out_channels=3,
            model_channels=32,
            attention_resolutions=[],
            num_res_blocks=1,
            channel_mult=[1],
            num_heads=1,
        )
        
        shape = (2, 3, 32, 32)
        samples = diffusion.p_sample_loop(model, shape, num_inference_steps=10)
        
        assert samples.shape == shape
        assert samples.dtype == torch.float32


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        rand1 = torch.randn(10)
        
        set_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        num_params = count_parameters(model)
        assert num_params == 55  # 10*5 + 5 bias terms


if __name__ == "__main__":
    pytest.main([__file__])
