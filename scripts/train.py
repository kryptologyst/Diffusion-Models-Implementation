#!/usr/bin/env python3
"""Main training script for diffusion models."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from src.diffusion_models.training.trainer import DiffusionTrainer
from src.diffusion_models.data.cifar10 import CIFAR10DataModule
from src.diffusion_models.utils import set_seed, get_device


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--gpus", type=int, help="Number of GPUs")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Precision")
    parser.add_argument("--overrides", nargs="*", help="Config overrides")
    
    return parser.parse_args()


def setup_logger(config: DictConfig) -> pl.loggers.Logger:
    """Setup logger based on config.
    
    Args:
        config: Configuration object.
        
    Returns:
        Logger instance.
    """
    if config.use_wandb:
        return WandbLogger(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"diffusion-{config.model.model_channels}ch",
            save_dir=config.log_dir,
        )
    else:
        return TensorBoardLogger(
            save_dir=config.log_dir,
            name="diffusion",
        )


def setup_callbacks(config: DictConfig) -> list:
    """Setup callbacks for training.
    
    Args:
        config: Configuration object.
        
    Returns:
        List of callbacks.
    """
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="diffusion-{epoch:02d}-{val_loss:.4f}",
        monitor=config.training.monitor,
        mode=config.training.mode,
        save_top_k=config.training.save_top_k,
        save_last=True,
        every_n_epochs=1,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=config.training.monitor,
        mode=config.training.mode,
        patience=50,
        verbose=True,
    )
    callbacks.append(early_stopping)
    
    return callbacks


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Apply overrides
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    
    # Override with command line arguments
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    
    # Set seed
    set_seed(config.seed)
    
    # Create directories
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Setup data module
    data_module = CIFAR10DataModule(**config.data)
    
    # Setup model
    model = DiffusionTrainer(
        model_config=config.model,
        diffusion_config=config.diffusion,
        training_config=config.training,
        evaluation_config=config.evaluation,
        use_wandb=config.use_wandb,
    )
    
    # Setup logger
    logger = setup_logger(config)
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Setup trainer
    trainer_kwargs = {
        "max_epochs": config.training.max_epochs,
        "precision": args.precision,
        "callbacks": callbacks,
        "logger": logger,
        "gradient_clip_val": config.training.gradient_clip_val,
        "accumulate_grad_batches": config.training.accumulate_grad_batches,
        "val_check_interval": config.training.val_check_interval,
        "log_every_n_steps": config.training.log_every_n_steps,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }
    
    # Add device-specific arguments
    if args.gpus is not None:
        trainer_kwargs["devices"] = args.gpus
    elif config.device == "auto":
        device = get_device("auto")
        if device.type == "cuda":
            trainer_kwargs["devices"] = "auto"
        else:
            trainer_kwargs["devices"] = 1
    else:
        trainer_kwargs["devices"] = 1
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.fit(model, data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
