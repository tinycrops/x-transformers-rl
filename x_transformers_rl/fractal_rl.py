"""
Fractal Reinforcement Learning Agent (FRALA)

Core concept: An agent that processes information and makes decisions based on a 
hierarchical, self-similar internal representation of its state or environment.
This representation mirrors the fractal concept where patterns repeat at different
scales, and a "global" aspect of the agent is consistent across these scales.

Key Components:
1. Fractal Encoder: Multi-level processing with self-similar blocks
2. Global State: Shared latent state across all fractal levels ("The Soul")
3. Inter-Level Transformations: Upscaling/downscaling between levels
4. Actor-Critic Integration: Compatible with x-transformers-rl framework

Inspired by HLIP's hierarchical attention and the fractal game concept.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict, Any
from functools import partial
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from x_transformers import (
    Decoder,
    ContinuousTransformerWrapper,
    Attention,
    FeedForward
)


class FractalLevelEmbedding(nn.Module):
    """Level-aware positional embedding for fractal levels"""
    
    def __init__(self, embed_dim: int, max_levels: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_levels = max_levels
        
        # Learnable level embeddings
        self.level_embeds = nn.Parameter(torch.randn(max_levels, embed_dim) * 0.02)
        
        # Scale embeddings (sinusoidal for different "zoom" levels)
        self.register_buffer('scale_embeds', self._create_scale_embeddings())
    
    def _create_scale_embeddings(self) -> Tensor:
        """Create sinusoidal embeddings for different scales"""
        position = torch.arange(self.max_levels).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / self.embed_dim))
        
        embeddings = torch.zeros(self.max_levels, self.embed_dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        
        return embeddings
    
    def forward(self, level_idx: int) -> Tensor:
        """Get embedding for a specific level"""
        level_embed = self.level_embeds[level_idx]
        scale_embed = self.scale_embeds[level_idx]
        return level_embed + scale_embed


class FractalProcessingBlock(nn.Module):
    """Self-similar processing block used at each fractal level"""
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_global_attention: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.use_global_attention = use_global_attention
        
        # Self-attention for local processing at this level
        self.self_attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        
        # Cross-attention to global state (if enabled)
        if use_global_attention:
            self.global_attn = Attention(
                dim=dim,
                dim_context=dim,  # Context dimension for cross-attention
                heads=heads,
                dim_head=dim_head,
                dropout=dropout
            )
        
        # Feed-forward network
        self.ff = FeedForward(
            dim=dim,
            mult=ff_mult,
            dropout=dropout
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) if use_global_attention else nn.Identity()
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(
        self, 
        x: Tensor, 
        global_state: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        # Self-attention
        x = self.norm1(x + self.self_attn(x, mask=mask))
        
        # Global attention (read from global state)
        if self.use_global_attention and global_state is not None:
            x = self.norm2(x + self.global_attn(x, context=global_state, mask=mask))
        
        # Feed-forward
        x = self.norm3(x + self.ff(x))
        
        return x


class FractalEncoder(nn.Module):
    """
    Core fractal encoder that processes information at multiple levels
    with self-similar processing blocks and global state management.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 512,
        num_levels: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.1,
        global_state_dim: Optional[int] = None,
        share_weights: bool = False,  # Whether to share weights across levels
        use_hypernetwork: bool = False  # Use hypernetwork for level-specific weights
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.share_weights = share_weights
        self.use_hypernetwork = use_hypernetwork
        
        # Global state dimension (defaults to embed_dim)
        self.global_state_dim = global_state_dim or embed_dim
        
        # Initial embedding
        self.input_embed = nn.Linear(input_dim, embed_dim)
        
        # Level embeddings
        self.level_embedding = FractalLevelEmbedding(embed_dim, num_levels)
        
        # Global state initialization and management
        self.global_state_init = nn.Parameter(torch.randn(1, 1, self.global_state_dim) * 0.02)
        self.global_state_update = nn.Linear(embed_dim, self.global_state_dim)
        
        # Fractal processing blocks
        if share_weights:
            # Single shared block
            self.fractal_block = FractalProcessingBlock(
                dim=embed_dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout
            )
        elif use_hypernetwork:
            # Hypernetwork to generate level-specific weights
            self.hypernet = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            self.base_block = FractalProcessingBlock(
                dim=embed_dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout
            )
        else:
            # Separate blocks for each level
            self.fractal_blocks = nn.ModuleList([
                FractalProcessingBlock(
                    dim=embed_dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout
                ) for _ in range(num_levels)
            ])
        
        # Inter-level transformation layers
        self.upscale_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_levels - 1)
        ])
        self.downscale_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_levels - 1)
        ])
        
        # Output projections for aggregation
        self.level_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_levels)
        ])
        
        # Final aggregation
        self.final_aggregation = nn.Sequential(
            nn.Linear(embed_dim * (num_levels + 1), embed_dim * 2),  # +1 for global state
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def upscale_level(self, fine_features: Tensor, num_sub_units: int, level_idx: int) -> Tensor:
        """
        Aggregate features from finer sub-units to a coarser level.
        Inspired by HLIP's _slice2scan operation.
        """
        batch_size, seq_len, dim = fine_features.shape
        
        # Reshape to group sub-units
        grouped = rearrange(fine_features, 'b (n s) d -> b n s d', s=num_sub_units)
        
        # Aggregate (mean pooling for now, could be learned)
        coarse_features = reduce(grouped, 'b n s d -> b n d', 'mean')
        
        # Apply level-specific transformation
        coarse_features = self.upscale_layers[level_idx](coarse_features)
        
        return coarse_features
    
    def downscale_level(self, coarse_features: Tensor, num_sub_units: int, level_idx: int) -> Tensor:
        """
        Distribute/refine features from a coarser level to finer sub-units.
        Inspired by HLIP's _study2slice operation.
        """
        batch_size, seq_len, dim = coarse_features.shape
        
        # Apply level-specific transformation
        transformed = self.downscale_layers[level_idx](coarse_features)
        
        # Expand to finer resolution
        fine_features = repeat(transformed, 'b n d -> b (n s) d', s=num_sub_units)
        
        return fine_features
    
    def get_fractal_block(self, level_idx: int) -> FractalProcessingBlock:
        """Get the processing block for a specific level"""
        if self.share_weights:
            return self.fractal_block
        elif self.use_hypernetwork:
            # Generate level-specific weights using hypernetwork
            level_embed = self.level_embedding(level_idx)
            level_weights = self.hypernet(level_embed)
            # For simplicity, we'll just use the base block with level conditioning
            # In a full implementation, you'd modify the block weights
            return self.base_block
        else:
            return self.fractal_blocks[level_idx]
    
    def forward(
        self, 
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_all_levels: bool = False
    ) -> Tensor | Tuple[Tensor, List[Tensor]]:
        """
        Forward pass through the fractal encoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            return_all_levels: Whether to return features from all levels
            
        Returns:
            Aggregated features or tuple of (aggregated, level_features)
        """
        batch_size = x.shape[0]
        
        # Initial embedding
        x = self.input_embed(x)
        
        # Initialize global state
        global_state = self.global_state_init.expand(batch_size, -1, -1)
        
        # Process through each fractal level
        level_outputs = []
        current_features = x
        
        for level_idx in range(self.num_levels):
            # Add level embedding
            level_embed = self.level_embedding(level_idx)
            level_features = current_features + level_embed.unsqueeze(0).unsqueeze(0)
            
            # Process through fractal block
            fractal_block = self.get_fractal_block(level_idx)
            level_features = fractal_block(level_features, global_state, mask)
            
            # Update global state based on level output
            global_update = reduce(level_features, 'b n d -> b 1 d', 'mean')
            global_update = self.global_state_update(global_update)
            global_state = global_state + global_update
            
            # Store level output
            level_outputs.append(level_features)
            
            # Prepare features for next level (if not last level)
            if level_idx < self.num_levels - 1:
                # For simplicity, we'll just pass features through
                # In practice, you might want to implement specific upscale/downscale logic
                current_features = level_features
        
        # Project each level's output
        projected_levels = []
        for i, level_output in enumerate(level_outputs):
            projected = self.level_projections[i](level_output)
            projected_levels.append(reduce(projected, 'b n d -> b d', 'mean'))
        
        # Include global state in aggregation
        global_state_flat = reduce(global_state, 'b n d -> b d', 'mean')
        
        # Concatenate all level features and global state
        all_features = torch.cat(projected_levels + [global_state_flat], dim=-1)
        
        # Final aggregation
        aggregated = self.final_aggregation(all_features)
        
        if return_all_levels:
            return aggregated, level_outputs
        
        return aggregated


class FractalWorldModelActorCritic(nn.Module):
    """
    Fractal-enabled version of WorldModelActorCritic that uses
    FractalEncoder instead of a standard transformer.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        critic_dim_pred: int,
        critic_min_max_value: Tuple[float, float],
        embed_dim: int = 512,
        num_fractal_levels: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.1,
        continuous_actions: bool = False,
        squash_continuous: bool = False,
        frac_actor_critic_head_gradient: float = 0.5,
        entropy_weight: float = 0.02,
        reward_dropout: float = 0.5,
        eps_clip: float = 0.2,
        value_clip: float = 0.4,
        evolutionary: bool = False,
        dim_latent_gene: Optional[int] = None,
        normalize_advantages: bool = True,
        fractal_share_weights: bool = False,
        fractal_use_hypernetwork: bool = False
    ):
        super().__init__()
        
        # Fractal encoder (replaces transformer)
        self.fractal_encoder = FractalEncoder(
            input_dim=state_dim,
            embed_dim=embed_dim,
            num_levels=num_fractal_levels,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            dropout=dropout,
            share_weights=fractal_share_weights,
            use_hypernetwork=fractal_use_hypernetwork
        )
        
        # Action and reward embeddings
        self.reward_embed = nn.Parameter(torch.ones(embed_dim) * 1e-2)
        
        if not continuous_actions:
            from x_transformers_rl.x_transformers_rl import SafeEmbedding
            self.action_embeds = SafeEmbedding(num_actions, embed_dim)
        else:
            self.action_embeds = nn.Linear(num_actions, embed_dim)
        
        self.reward_dropout = nn.Dropout(reward_dropout)
        
        # State embedding
        self.to_state_embed = nn.Linear(state_dim, embed_dim)
        
        # World modeling heads
        self.to_pred_done = nn.Sequential(
            nn.Linear(embed_dim * 2, 1),
            Rearrange('... 1 -> ...'),
            nn.Sigmoid()
        )
        
        state_dim_and_reward = state_dim + 1
        
        from x_transformers_rl.x_transformers_rl import Continuous
        self.to_pred = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            Continuous.Linear(embed_dim, state_dim_and_reward)
        )
        
        # Evolutionary support
        self.evolutionary = evolutionary
        if evolutionary:
            assert dim_latent_gene is not None
            self.latent_to_embed = nn.Linear(dim_latent_gene, embed_dim)
        
        # Actor-Critic heads
        actor_critic_input_dim = embed_dim * 2
        if evolutionary:
            actor_critic_input_dim += embed_dim
        
        # Critic head with HL-Gauss loss
        from hl_gauss_pytorch import HLGaussLoss
        self.critic_head = nn.Sequential(
            nn.Linear(actor_critic_input_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, critic_dim_pred)
        )
        
        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value=critic_min_max_value[0],
            max_value=critic_min_max_value[1],
            num_bins=critic_dim_pred,
            clamp_to_range=True
        )
        
        # Actor head
        from x_transformers_rl.x_transformers_rl import Discrete
        action_type_klass = Discrete if not continuous_actions else Continuous
        
        self.action_head = nn.Sequential(
            nn.Linear(actor_critic_input_dim, embed_dim * 2),
            nn.SiLU(),
            action_type_klass.Linear(embed_dim * 2, num_actions)
        )
        
        if continuous_actions and squash_continuous:
            action_type_klass = partial(action_type_klass, squash=True)
        
        self.action_type_klass = action_type_klass
        self.squash_continuous = squash_continuous and continuous_actions
        
        # Training parameters
        self.frac_actor_critic_head_gradient = frac_actor_critic_head_gradient
        self.eps_clip = eps_clip
        self.entropy_weight = entropy_weight
        self.value_clip = value_clip
        
        # Advantage normalization
        from x_transformers_rl.x_transformers_rl import normalize, identity
        self.maybe_normalize = normalize if normalize_advantages else identity
        
        self.register_buffer('dummy', torch.tensor(0), persistent=False)
    
    @property
    def device(self):
        return self.dummy.device
    
    def compute_actor_loss(
        self,
        raw_actions: Tensor,
        actions: Tensor,
        old_log_probs: Tensor,
        returns: Tensor,
        old_values: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Compute PPO actor loss"""
        dist = self.action_type_klass(raw_actions)
        action_log_probs = dist.log_prob(actions)
        
        entropy = dist.entropy() if not self.squash_continuous else -action_log_probs
        
        scalar_old_values = self.critic_hl_gauss_loss(old_values)
        
        # PPO clipped surrogate loss
        ratios = (action_log_probs - old_log_probs).exp()
        clipped_ratios = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip)
        
        advantages = returns - scalar_old_values.detach()
        maybe_normed_advantages = self.maybe_normalize(advantages, mask=mask)
        
        from einx import multiply
        surr1 = multiply('b n ..., b n -> b n ...', ratios, maybe_normed_advantages)
        surr2 = multiply('b n ..., b n -> b n ...', clipped_ratios, maybe_normed_advantages)
        
        actor_loss = -torch.min(surr1, surr2) - self.entropy_weight * entropy
        actor_loss = reduce(actor_loss, 'b n ... -> b n', 'sum')
        
        return actor_loss
    
    def compute_critic_loss(
        self,
        values: Tensor,
        returns: Tensor,
        old_values: Tensor
    ) -> Tensor:
        """Compute clipped value loss"""
        clip, hl_gauss = self.value_clip, self.critic_hl_gauss_loss
        
        scalar_old_values = hl_gauss(old_values)
        scalar_values = hl_gauss(values)
        
        clipped_returns = returns.clamp(-clip, clip)
        
        clipped_loss = hl_gauss(values, clipped_returns)
        loss = hl_gauss(values, returns)
        
        old_values_lo = scalar_old_values - clip
        old_values_hi = scalar_old_values + clip
        
        def is_between(mid, lo, hi):
            return (lo < mid) & (mid < hi)
        
        critic_loss = torch.where(
            is_between(scalar_values, returns, old_values_lo) |
            is_between(scalar_values, old_values_hi, returns),
            0.,
            torch.min(loss, clipped_loss)
        )
        
        return critic_loss
    
    def forward(
        self,
        state: Tensor,
        actions: Optional[Tensor] = None,
        rewards: Optional[Tensor] = None,
        next_actions: Optional[Tensor] = None,
        latent_gene: Optional[Tensor] = None,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Dict[str, Any]]:
        """
        Forward pass through fractal world model actor-critic.
        
        Returns:
            raw_actions: Raw action predictions
            values: Value predictions
            state_pred: Next state predictions (if next_actions provided)
            dones: Done predictions (if next_actions provided)
            cache: Additional info (fractal level outputs, etc.)
        """
        device = self.device
        batch_size = state.shape[0]
        
        # Process through fractal encoder
        fractal_features, level_outputs = self.fractal_encoder(
            state, return_all_levels=True
        )
        
        # State embedding for actor-critic input
        state_embed = self.to_state_embed(state)
        
        # World model predictions (if next_actions provided)
        state_pred = None
        dones = None
        
        if next_actions is not None:
            next_action_embeds = self.action_embeds(next_actions)
            embed_with_actions = torch.cat((fractal_features, next_action_embeds), dim=-1)
            
            from x_transformers_rl.x_transformers_rl import Continuous
            raw_state_pred = self.to_pred(embed_with_actions)
            state_pred = Continuous(raw_state_pred).mean_variance
            dones = self.to_pred_done(embed_with_actions)
        
        # Actor-critic processing
        from x_transformers_rl.x_transformers_rl import frac_gradient
        fractal_features = frac_gradient(fractal_features, self.frac_actor_critic_head_gradient)
        
        # Actor-critic input
        if state_embed.ndim == 3:
            state_embed = reduce(state_embed, 'b n d -> b d', 'mean')
        
        actor_critic_input = torch.cat((fractal_features, state_embed), dim=-1)
        
        # Evolutionary conditioning
        if self.evolutionary and latent_gene is not None:
            latent_embed = self.latent_to_embed(latent_gene)
            if latent_embed.ndim == 2 and actor_critic_input.ndim == 3:
                latent_embed = repeat(latent_embed, 'b d -> b n d', n=actor_critic_input.shape[1])
            actor_critic_input = torch.cat((actor_critic_input, latent_embed), dim=-1)
        
        # Generate actions and values
        raw_actions = self.action_head(actor_critic_input)
        values = self.critic_head(actor_critic_input)
        
        # Prepare cache with fractal-specific info
        cache = {
            'fractal_levels': level_outputs,
            'global_state': None,  # Could extract from fractal_encoder if needed
            'level_features': [reduce(level, 'b n d -> b d', 'mean') for level in level_outputs]
        }
        
        return raw_actions, values, state_pred, dones, cache


def create_fractal_agent(
    state_dim: int,
    num_actions: int,
    reward_range: Tuple[float, float],
    continuous_actions: bool = False,
    num_fractal_levels: int = 4,
    embed_dim: int = 512,
    **kwargs
) -> Any:
    """
    Create a fractal RL agent using the x-transformers-rl Agent class
    but with FractalWorldModelActorCritic as the world model.
    """
    from x_transformers_rl.x_transformers_rl import Agent
    
    # Create fractal world model
    fractal_world_model = FractalWorldModelActorCritic(
        state_dim=state_dim,
        num_actions=num_actions,
        critic_dim_pred=kwargs.get('critic_pred_num_bins', 100),
        critic_min_max_value=reward_range,
        embed_dim=embed_dim,
        num_fractal_levels=num_fractal_levels,
        continuous_actions=continuous_actions,
        **{k: v for k, v in kwargs.items() if k.startswith('fractal_')}
    )
    
    # Create agent with fractal world model
    # Note: This would require modifying the Agent class to accept a custom world model
    # For now, this is a conceptual example
    
    return {
        'fractal_world_model': fractal_world_model,
        'state_dim': state_dim,
        'num_actions': num_actions,
        'continuous_actions': continuous_actions,
        'num_fractal_levels': num_fractal_levels
    }


# Example usage and testing functions
def test_fractal_encoder():
    """Test the fractal encoder with dummy data"""
    batch_size, seq_len, input_dim = 4, 16, 128
    
    encoder = FractalEncoder(
        input_dim=input_dim,
        embed_dim=256,
        num_levels=3,
        heads=8
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test forward pass
    output = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with return_all_levels
    output, level_outputs = encoder(x, return_all_levels=True)
    print(f"Number of levels: {len(level_outputs)}")
    for i, level_out in enumerate(level_outputs):
        print(f"Level {i} output shape: {level_out.shape}")


def test_fractal_world_model():
    """Test the fractal world model actor-critic"""
    batch_size, seq_len, state_dim = 4, 16, 64
    num_actions = 8
    
    model = FractalWorldModelActorCritic(
        state_dim=state_dim,
        num_actions=num_actions,
        critic_dim_pred=100,
        critic_min_max_value=(-10.0, 10.0),
        embed_dim=256,
        num_fractal_levels=3
    )
    
    state = torch.randn(batch_size, seq_len, state_dim)
    
    # Test forward pass
    raw_actions, values, state_pred, dones, cache = model(state)
    
    print(f"State shape: {state.shape}")
    print(f"Raw actions shape: {raw_actions.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Cache keys: {cache.keys()}")
    print(f"Number of fractal levels in cache: {len(cache['fractal_levels'])}")


if __name__ == "__main__":
    print("Testing Fractal Encoder...")
    test_fractal_encoder()
    print("\nTesting Fractal World Model...")
    test_fractal_world_model()
    print("\nFRALA implementation complete!") 