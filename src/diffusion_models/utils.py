"""Core utilities for diffusion models."""

import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Get the best available device.
    
    Args:
        device: Device specification. If "auto", automatically select best device.
        
    Returns:
        PyTorch device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object.
        save_path: Path to save configuration.
    """
    OmegaConf.save(config, save_path)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> str:
    """Get human-readable model size.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Model size as string (e.g., "1.2M", "45.6M").
    """
    num_params = count_parameters(model)
    
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f}K"
    else:
        return str(num_params)


def create_checkpoint_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    loss: float,
    config: DictConfig,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create a checkpoint dictionary.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        epoch: Current epoch.
        step: Current step.
        loss: Current loss value.
        config: Configuration object.
        **kwargs: Additional items to include in checkpoint.
        
    Returns:
        Checkpoint dictionary.
    """
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    checkpoint.update(kwargs)
    return checkpoint
