#!/usr/bin/env python3
"""Setup script for diffusion models project."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Diffusion Models Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create necessary directories
    directories = [
        "data", "checkpoints", "logs", "assets", "assets/generated"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Install pre-commit hooks
    if run_command("pre-commit install", "Installing pre-commit hooks"):
        print("✅ Pre-commit hooks installed")
    else:
        print("⚠️  Pre-commit hooks installation failed (optional)")
    
    # Download CIFAR-10 dataset
    print("🔄 Downloading CIFAR-10 dataset...")
    try:
        import torchvision.datasets as datasets
        datasets.CIFAR10(root="./data", train=True, download=True)
        datasets.CIFAR10(root="./data", train=False, download=True)
        print("✅ CIFAR-10 dataset downloaded")
    except Exception as e:
        print(f"⚠️  Dataset download failed: {e}")
        print("   Dataset will be downloaded automatically during training")
    
    # Run tests
    if run_command("python -m pytest tests/ -v", "Running tests"):
        print("✅ All tests passed")
    else:
        print("⚠️  Some tests failed (check implementation)")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train a model: python scripts/train.py")
    print("2. Evaluate model: python scripts/evaluate.py --checkpoint checkpoints/best.ckpt")
    print("3. Launch demo: streamlit run demo/streamlit_app.py")
    print("4. Run example: python 0372.py")


if __name__ == "__main__":
    main()
