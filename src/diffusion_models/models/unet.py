"""UNet model for diffusion models."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim: int):
        """Initialize sinusoidal embeddings.
        
        Args:
            dim: Embedding dimension.
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            time: Timestep tensor.
            
        Returns:
            Position embeddings.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embeddings."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        """Initialize residual block.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            time_emb_dim: Time embedding dimension.
            dropout: Dropout rate.
            use_scale_shift_norm: Whether to use scale-shift normalization.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Main convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * out_channels if use_scale_shift_norm else out_channels),
        )
        
        # Normalization layers
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            time_emb: Time embeddings.
            
        Returns:
            Output tensor.
        """
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Apply time embeddings
        time_emb = self.time_mlp(time_emb)
        if self.use_scale_shift_norm:
            scale, shift = time_emb.chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        else:
            h = h + time_emb[:, :, None, None]
        
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Skip connection
        return h + self.skip_connection(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        """Initialize attention block.
        
        Args:
            channels: Number of channels.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        b, c, h, w = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # Compute QKV
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = rearrange(q, "b (h d) h2 w2 -> b h (h2 w2) d", h=self.num_heads)
        k = rearrange(k, "b (h d) h2 w2 -> b h (h2 w2) d", h=self.num_heads)
        v = rearrange(v, "b (h d) h2 w2 -> b h (h2 w2) d", h=self.num_heads)
        
        # Compute attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.einsum("bhqd,bhkd->bhqk", q, k) * scale, dim=-1)
        out = torch.einsum("bhqk,bhvd->bhqd", attn, v)
        
        # Reshape back
        out = rearrange(out, "b h (h2 w2) d -> b (h d) h2 w2", h2=h, w2=w)
        
        # Project and add residual
        out = self.proj(out)
        return x + out


class Downsample(nn.Module):
    """Downsampling layer."""
    
    def __init__(self, channels: int):
        """Initialize downsampling layer.
        
        Args:
            channels: Number of channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Downsampled tensor.
        """
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer."""
    
    def __init__(self, channels: int):
        """Initialize upsampling layer.
        
        Args:
            channels: Number of channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Upsampled tensor.
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """UNet model for diffusion models."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        attention_resolutions: List[int] = [16, 8],
        num_res_blocks: int = 2,
        channel_mult: List[int] = [1, 2, 2, 2],
        num_heads: int = 4,
        use_scale_shift_norm: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize UNet model.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            model_channels: Base number of channels.
            attention_resolutions: Resolutions to apply attention.
            num_res_blocks: Number of residual blocks per resolution.
            channel_mult: Channel multipliers for each resolution.
            num_heads: Number of attention heads.
            use_scale_shift_norm: Whether to use scale-shift normalization.
            dropout: Dropout rate.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.attention_resolutions = attention_resolutions
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_scale_shift_norm = use_scale_shift_norm
        self.dropout = dropout
        
        # Time embeddings
        time_emb_dim = model_channels * 4
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, model_channels * mult, time_emb_dim, dropout, use_scale_shift_norm)]
                ch = model_channels * mult
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                
                self.down_blocks.append(nn.Sequential(*layers))
            
            if level != len(channel_mult) - 1:
                self.down_samples.append(Downsample(ch))
                ds *= 2
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle block
        self.middle_block1 = ResidualBlock(ch, ch, time_emb_dim, dropout, use_scale_shift_norm)
        self.middle_attn = AttentionBlock(ch, num_heads)
        self.middle_block2 = ResidualBlock(ch, ch, time_emb_dim, dropout, use_scale_shift_norm)
        
        # Upsampling path
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResidualBlock(ch + model_channels * mult, model_channels * mult, time_emb_dim, dropout, use_scale_shift_norm)]
                ch = model_channels * mult
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                
                self.up_blocks.append(nn.Sequential(*layers))
            
            if level != 0:
                self.up_samples.append(Upsample(ch))
                ds //= 2
            else:
                self.up_samples.append(nn.Identity())
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            timesteps: Timestep tensor.
            
        Returns:
            Output tensor.
        """
        # Time embeddings
        t_emb = self.time_emb(timesteps)
        
        # Input projection
        h = self.input_proj(x)
        
        # Store skip connections
        hs = [h]
        
        # Downsampling path
        for i, (down_block, down_sample) in enumerate(zip(self.down_blocks, self.down_samples)):
            h = down_block[0](h, t_emb)
            if len(down_block) > 1:
                h = down_block[1](h)
            hs.append(h)
            h = down_sample(h)
        
        # Middle block
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)
        
        # Upsampling path
        for up_block, up_sample in zip(self.up_blocks, self.up_samples):
            h = torch.cat([h, hs.pop()], dim=1)
            h = up_block[0](h, t_emb)
            if len(up_block) > 1:
                h = up_block[1](h)
            h = up_sample(h)
        
        # Output projection
        return self.output_proj(h)
