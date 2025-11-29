"""Streamlit demo for diffusion models."""

import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from src.diffusion_models.sampling.sampler import DiffusionSampler, load_model_from_checkpoint
from src.diffusion_models.utils import get_device, set_seed


def load_model():
    """Load the diffusion model."""
    if "model" not in st.session_state:
        # Check if checkpoint exists
        checkpoint_path = "checkpoints/last.ckpt"
        config_path = "configs/config.yaml"
        
        if os.path.exists(checkpoint_path) and os.path.exists(config_path):
            try:
                model, scheduler = load_model_from_checkpoint(
                    checkpoint_path,
                    config_path,
                    "auto",
                )
                sampler = DiffusionSampler(model, scheduler, "auto")
                st.session_state.model = sampler
                st.session_state.model_loaded = True
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.session_state.model_loaded = False
        else:
            st.warning("Model checkpoint not found. Please train a model first.")
            st.session_state.model_loaded = False


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Diffusion Models Demo",
        page_icon="🎨",
        layout="wide",
    )
    
    st.title("🎨 Diffusion Models Demo")
    st.markdown("Generate images using diffusion models trained on CIFAR-10")
    
    # Load model
    load_model()
    
    if not st.session_state.get("model_loaded", False):
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("Generation Controls")
    
    # Number of samples
    num_samples = st.sidebar.slider(
        "Number of samples",
        min_value=1,
        max_value=64,
        value=16,
        step=1,
    )
    
    # Number of inference steps
    num_steps = st.sidebar.slider(
        "Number of inference steps",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
    )
    
    # Stochasticity parameter
    eta = st.sidebar.slider(
        "Stochasticity (eta)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
    )
    
    # Random seed
    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=2**32 - 1,
        value=42,
        step=1,
    )
    
    # Generate button
    if st.sidebar.button("Generate Samples", type="primary"):
        with st.spinner("Generating samples..."):
            # Generate samples
            samples = st.session_state.model.sample(
                num_samples=num_samples,
                num_inference_steps=num_steps,
                eta=eta,
                seed=seed,
            )
            
            # Convert to numpy and denormalize
            samples_np = samples.cpu().numpy()
            samples_np = (samples_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            samples_np = np.clip(samples_np, 0, 1)
            
            # Store in session state
            st.session_state.generated_samples = samples_np
    
    # Display generated samples
    if "generated_samples" in st.session_state:
        st.header("Generated Samples")
        
        samples = st.session_state.generated_samples
        
        # Create grid
        n_cols = 4
        n_rows = (len(samples) + n_cols - 1) // n_cols
        
        for row in range(n_rows):
            cols = st.columns(n_cols)
            for col_idx in range(n_cols):
                sample_idx = row * n_cols + col_idx
                if sample_idx < len(samples):
                    with cols[col_idx]:
                        # Convert to PIL Image
                        img = Image.fromarray((samples[sample_idx].transpose(1, 2, 0) * 255).astype(np.uint8))
                        st.image(img, caption=f"Sample {sample_idx + 1}")
        
        # Download button
        if st.button("Download Samples"):
            # Create a grid image
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
            if n_rows == 1:
                axes = [axes]
            if n_cols == 1:
                axes = [[ax] for ax in axes]
            
            for row in range(n_rows):
                for col in range(n_cols):
                    sample_idx = row * n_cols + col
                    if sample_idx < len(samples):
                        axes[row][col].imshow(samples[sample_idx].transpose(1, 2, 0))
                        axes[row][col].axis("off")
                    else:
                        axes[row][col].axis("off")
            
            plt.tight_layout()
            plt.savefig("generated_samples.png", dpi=150, bbox_inches="tight")
            plt.close()
            
            with open("generated_samples.png", "rb") as f:
                st.download_button(
                    label="Download Grid",
                    data=f.read(),
                    file_name="generated_samples.png",
                    mime="image/png",
                )
    
    # Interpolation section
    st.header("Interpolation")
    st.markdown("Generate interpolated samples between two random points in the latent space")
    
    col1, col2 = st.columns(2)
    
    with col1:
        interp_steps = st.slider(
            "Interpolation steps",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
        )
    
    with col2:
        interp_seed = st.number_input(
            "Interpolation seed",
            min_value=0,
            max_value=2**32 - 1,
            value=123,
            step=1,
        )
    
    if st.button("Generate Interpolation"):
        with st.spinner("Generating interpolation..."):
            # Generate interpolation
            interp_samples = st.session_state.model.interpolate(
                num_steps=interp_steps,
                num_inference_steps=num_steps,
                eta=eta,
                seed=interp_seed,
            )
            
            # Convert to numpy and denormalize
            interp_samples_np = interp_samples.cpu().numpy()
            interp_samples_np = (interp_samples_np + 1) / 2
            interp_samples_np = np.clip(interp_samples_np, 0, 1)
            
            # Store in session state
            st.session_state.interpolation_samples = interp_samples_np
    
    # Display interpolation
    if "interpolation_samples" in st.session_state:
        st.header("Interpolation Results")
        
        interp_samples = st.session_state.interpolation_samples
        
        # Display as horizontal strip
        cols = st.columns(len(interp_samples))
        for i, col in enumerate(cols):
            with col:
                img = Image.fromarray((interp_samples[i].transpose(1, 2, 0) * 255).astype(np.uint8))
                st.image(img, caption=f"Step {i + 1}")
        
        # Download interpolation
        if st.button("Download Interpolation"):
            fig, ax = plt.subplots(1, len(interp_samples), figsize=(len(interp_samples) * 2, 2))
            for i, sample in enumerate(interp_samples):
                ax[i].imshow(sample.transpose(1, 2, 0))
                ax[i].axis("off")
                ax[i].set_title(f"Step {i + 1}")
            
            plt.tight_layout()
            plt.savefig("interpolation.png", dpi=150, bbox_inches="tight")
            plt.close()
            
            with open("interpolation.png", "rb") as f:
                st.download_button(
                    label="Download Interpolation",
                    data=f.read(),
                    file_name="interpolation.png",
                    mime="image/png",
                )
    
    # Model info
    st.sidebar.header("Model Info")
    st.sidebar.info("""
    This demo uses a UNet-based diffusion model trained on CIFAR-10.
    
    **Parameters:**
    - Model channels: 128
    - Attention resolutions: [16, 8]
    - Number of residual blocks: 2
    - Channel multipliers: [1, 2, 2, 2]
    - Number of attention heads: 4
    """)
    
    # Tips
    st.sidebar.header("Tips")
    st.sidebar.info("""
    - Higher inference steps = better quality but slower generation
    - eta=0 for deterministic sampling (DDIM)
    - eta=1 for stochastic sampling (DDPM)
    - Use different seeds to generate different samples
    """)


if __name__ == "__main__":
    main()
