# Diffusion Models Implementation

A production-ready implementation of diffusion models for image generation, featuring a UNet architecture trained on CIFAR-10.

## Features

- **Modern Architecture**: UNet-based diffusion model with attention mechanisms
- **Flexible Training**: PyTorch Lightning integration with comprehensive logging
- **Evaluation Metrics**: FID, Inception Score, and LPIPS diversity
- **Interactive Demo**: Streamlit-based web interface
- **Production Ready**: Proper configuration management, checkpointing, and device support
- **Reproducible**: Deterministic seeding and comprehensive documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Diffusion-Models-Implementation.git
cd Diffusion-Models-Implementation

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config configs/config.yaml

# Resume from checkpoint
python scripts/train.py --resume checkpoints/last.ckpt

# Train with specific device
python scripts/train.py --device cuda --gpus 1
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint checkpoints/best.ckpt

# Evaluate with specific metrics
python scripts/evaluate.py --checkpoint checkpoints/best.ckpt --compute_fid --compute_is --compute_lpips
```

### Sampling

```bash
# Generate samples
python -m src.diffusion_models.sampling.sampler --checkpoint checkpoints/best.ckpt --output samples.png

# Generate interpolation
python -m src.diffusion_models.sampling.sampler --checkpoint checkpoints/best.ckpt --interpolate --output interpolation.png
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/streamlit_app.py
```

## Project Structure

```
diffusion-models-implementation/
├── src/diffusion_models/          # Main package
│   ├── models/                   # Model architectures
│   │   └── unet.py              # UNet implementation
│   ├── data/                    # Data modules
│   │   └── cifar10.py          # CIFAR-10 data module
│   ├── training/               # Training components
│   │   └── trainer.py          # PyTorch Lightning trainer
│   ├── evaluation/              # Evaluation metrics
│   │   └── evaluator.py        # Model evaluator
│   ├── sampling/                # Sampling utilities
│   │   └── sampler.py          # Sample generation
│   ├── diffusion.py            # Diffusion process
│   └── utils.py                # Utility functions
├── configs/                     # Configuration files
│   └── config.yaml             # Main configuration
├── scripts/                     # Training and evaluation scripts
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── demo/                       # Interactive demos
│   └── streamlit_app.py        # Streamlit demo
├── tests/                      # Unit tests
├── assets/                     # Generated samples and visualizations
├── checkpoints/                # Model checkpoints
├── logs/                       # Training logs
└── data/                       # Dataset storage
```

## Configuration

The project uses OmegaConf for configuration management. Key configuration options:

### Model Configuration
- `model_channels`: Base number of channels (default: 128)
- `attention_resolutions`: Resolutions to apply attention (default: [16, 8])
- `num_res_blocks`: Number of residual blocks per resolution (default: 2)
- `channel_mult`: Channel multipliers for each resolution (default: [1, 2, 2, 2])
- `num_heads`: Number of attention heads (default: 4)

### Diffusion Configuration
- `num_timesteps`: Number of diffusion timesteps (default: 1000)
- `beta_schedule`: Type of beta schedule (linear/cosine)
- `prediction_type`: Type of prediction (epsilon/v_prediction/sample)
- `loss_type`: Loss function type (mse/l1/huber)

### Training Configuration
- `max_epochs`: Maximum number of training epochs
- `learning_rate`: Learning rate for optimizer
- `batch_size`: Training batch size
- `gradient_clip_val`: Gradient clipping value

## Model Architecture

The implementation features a modern UNet architecture with:

- **Residual Blocks**: Skip connections with time embeddings
- **Attention Mechanisms**: Self-attention at multiple resolutions
- **Time Embeddings**: Sinusoidal position embeddings for timesteps
- **Scale-Shift Normalization**: Conditional normalization with time information
- **Progressive Downsampling**: Multi-scale feature extraction

## Training Features

- **PyTorch Lightning**: Modern training framework with automatic optimization
- **Mixed Precision**: Automatic mixed precision training for efficiency
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Checkpointing**: Automatic model saving and resuming
- **Logging**: TensorBoard and Weights & Biases integration
- **Early Stopping**: Prevents overfitting

## Evaluation Metrics

- **FID (Fréchet Inception Distance)**: Measures quality and diversity
- **IS (Inception Score)**: Measures quality and diversity
- **LPIPS Diversity**: Measures perceptual diversity
- **Sample Quality**: Visual inspection of generated samples

## Sampling Methods

- **DDPM**: Stochastic sampling with full noise schedule
- **DDIM**: Deterministic sampling with reduced steps
- **Interpolation**: Latent space interpolation between samples
- **Progressive Sampling**: Step-by-step denoising visualization

## Device Support

- **CUDA**: NVIDIA GPU acceleration
- **MPS**: Apple Silicon GPU acceleration
- **CPU**: Fallback CPU computation
- **Auto-detection**: Automatic device selection

## Development

### Code Quality

The project follows modern Python development practices:

- **Type Hints**: Comprehensive type annotations
- **Documentation**: Google-style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff for code quality
- **Testing**: Pytest for unit tests

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Performance

### Training Performance
- **CIFAR-10**: ~2-3 hours on RTX 3080
- **Memory Usage**: ~8GB VRAM with batch size 128
- **Convergence**: Typically converges in 200-500 epochs

### Generation Performance
- **DDPM (1000 steps)**: ~2 seconds per sample
- **DDIM (50 steps)**: ~0.1 seconds per sample
- **Batch Generation**: Efficient parallel sampling

## Limitations

- **Dataset**: Currently only supports CIFAR-10 (32x32 images)
- **Architecture**: Fixed UNet architecture (not configurable)
- **Conditioning**: No text or class conditioning implemented
- **Resolution**: Limited to 32x32 pixel images

## Future Improvements

- **Higher Resolution**: Support for 64x64 and 128x128 images
- **Text Conditioning**: CLIP-based text-to-image generation
- **Class Conditioning**: Class-conditional generation
- **Latent Diffusion**: VAE-based latent space diffusion
- **ControlNet**: Controllable generation with additional inputs

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{diffusion_models_implementation,
  title={Diffusion Models Implementation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Diffusion-Models-Implementation}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al.
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) - Nichol & Dhariwal
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) - Song et al.
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/) - Lightning team
- [Diffusers](https://huggingface.co/docs/diffusers/) - Hugging Face team
# Diffusion-Models-Implementation
