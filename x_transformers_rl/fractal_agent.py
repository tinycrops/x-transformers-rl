"""
FRALA Agent - Fractal Reinforcement Learning Agent

This module extends the x-transformers-rl Agent class to use fractal processing
instead of the standard transformer architecture. It integrates the FractalWorldModelActorCritic
with the existing RL training loop and provides a complete fractal agent implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Dict, Any, List
from functools import partial

from x_transformers_rl.x_transformers_rl import (
    Agent,
    WorldModelActorCritic,
    Memory,
    create_memory,
    calc_gae,
    normalize,
    identity,
    exists,
    default,
    first,
    pad_at_dim,
    from_numpy,
    RSNorm
)

from x_transformers_rl.fractal_rl import (
    FractalWorldModelActorCritic,
    FractalEncoder
)

from x_transformers import ContinuousTransformerWrapper
from einops import reduce, repeat, rearrange
from accelerate import Accelerator
import numpy as np


class FractalAgent(Agent):
    """
    Fractal Reinforcement Learning Agent that extends the base Agent class
    to use fractal processing instead of standard transformers.
    
    Key differences from base Agent:
    1. Uses FractalWorldModelActorCritic instead of WorldModelActorCritic
    2. Supports fractal-specific parameters and configurations
    3. Maintains compatibility with existing training loop and evolution
    """
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        reward_range: Tuple[float, float],
        epochs: int,
        max_timesteps: int,
        batch_size: int,
        lr: float,
        betas: Tuple[float, float],
        lam: float,
        gamma: float,
        beta_s: float,
        regen_reg_rate: float,
        cautious_factor: float,
        eps_clip: float,
        value_clip: float,
        ema_decay: float,
        continuous_actions: bool = False,
        squash_continuous: bool = True,
        critic_pred_num_bins: int = 100,
        hidden_dim: int = 48,
        evolutionary: bool = False,
        evolve_every: int = 1,
        evolve_after_step: int = 20,
        latent_gene_pool: dict = dict(
            dim=128,
            num_genes_per_island=3,
            num_selected=2,
            tournament_size=2
        ),
        # Fractal-specific parameters
        num_fractal_levels: int = 4,
        fractal_embed_dim: int = 256,
        fractal_heads: int = 8,
        fractal_dim_head: int = 32,
        fractal_ff_mult: int = 4,
        fractal_share_weights: bool = False,
        fractal_use_hypernetwork: bool = False,
        fractal_global_state_dim: Optional[int] = None,
        # Traditional world model parameters (for compatibility)
        world_model: dict = dict(
            attn_dim_head=16,
            heads=4,
            depth=4,
            attn_gate_values=True,
            add_value_residual=True,
            learned_value_residual_mix=True
        ),
        dropout: float = 0.25,
        max_grad_norm: float = 0.5,
        frac_actor_critic_head_gradient: float = 0.5,
        ema_kwargs: dict = dict(
            update_model_with_ema_every=1250
        ),
        save_path: str = './fractal_ppo.pt',
        accelerator: Optional[Accelerator] = None,
        actor_loss_weight: float = 1.,
        critic_loss_weight: float = 1.,
        autoregressive_loss_weight: float = 1.
    ):
        # Store parameters needed before calling super().__init__
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.reward_range = reward_range
        self.continuous_actions = continuous_actions
        self.squash_continuous = squash_continuous
        self.critic_pred_num_bins = critic_pred_num_bins
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.evolutionary = evolutionary
        self.dropout = dropout
        self.frac_actor_critic_head_gradient = frac_actor_critic_head_gradient
        self.lr = lr
        self.betas = betas
        self.accelerator = accelerator
        
        # Store fractal-specific parameters
        self.num_fractal_levels = num_fractal_levels
        self.fractal_embed_dim = fractal_embed_dim
        self.fractal_heads = fractal_heads
        self.fractal_dim_head = fractal_dim_head
        self.fractal_ff_mult = fractal_ff_mult
        self.fractal_share_weights = fractal_share_weights
        self.fractal_use_hypernetwork = fractal_use_hypernetwork
        self.fractal_global_state_dim = fractal_global_state_dim
        
        # Store gene pool info before calling super
        if evolutionary:
            from x_transformers_rl.evolution import LatentGenePool
            gene_pool = LatentGenePool(**latent_gene_pool)
            self.dim_latent_gene = gene_pool.dim_gene
        else:
            self.dim_latent_gene = None

        # Initialize base Agent class
        super().__init__(
            state_dim=state_dim,
            num_actions=num_actions,
            reward_range=reward_range,
            epochs=epochs,
            max_timesteps=max_timesteps,
            batch_size=batch_size,
            lr=lr,
            betas=betas,
            lam=lam,
            gamma=gamma,
            beta_s=beta_s,
            regen_reg_rate=regen_reg_rate,
            cautious_factor=cautious_factor,
            eps_clip=eps_clip,
            value_clip=value_clip,
            ema_decay=ema_decay,
            continuous_actions=continuous_actions,
            squash_continuous=squash_continuous,
            critic_pred_num_bins=critic_pred_num_bins,
            hidden_dim=hidden_dim,
            evolutionary=evolutionary,
            evolve_every=evolve_every,
            evolve_after_step=evolve_after_step,
            latent_gene_pool=latent_gene_pool,
            world_model=world_model,  # This will be overridden
            dropout=dropout,
            max_grad_norm=max_grad_norm,
            frac_actor_critic_head_gradient=frac_actor_critic_head_gradient,
            ema_kwargs=ema_kwargs,
            save_path=save_path,
            accelerator=accelerator,
            actor_loss_weight=actor_loss_weight,
            critic_loss_weight=critic_loss_weight,
            autoregressive_loss_weight=autoregressive_loss_weight
        )
        
        # Replace the world model with fractal version
        self._create_fractal_world_model()
    
    def _create_fractal_world_model(self):
        """Create and initialize the fractal world model, replacing the standard one."""
        
        # Create fractal world model
        self.world_model = FractalWorldModelActorCritic(
            state_dim=self.state_dim,
            num_actions=self.num_actions,
            critic_dim_pred=self.critic_pred_num_bins,
            critic_min_max_value=self.reward_range,
            embed_dim=self.fractal_embed_dim,
            num_fractal_levels=self.num_fractal_levels,
            heads=self.fractal_heads,
            dim_head=self.fractal_dim_head,
            ff_mult=self.fractal_ff_mult,
            dropout=self.dropout,
            continuous_actions=self.continuous_actions,
            squash_continuous=self.squash_continuous,
            frac_actor_critic_head_gradient=self.frac_actor_critic_head_gradient,
            eps_clip=self.eps_clip,
            value_clip=self.value_clip,
            evolutionary=self.evolutionary,
            dim_latent_gene=self.dim_latent_gene if self.evolutionary else None,
            fractal_share_weights=self.fractal_share_weights,
            fractal_use_hypernetwork=self.fractal_use_hypernetwork
        )
        
        # Update optimizer to include fractal world model parameters
        if hasattr(self, 'optimizer'):
            from adam_atan2_pytorch import AdoptAtan2
            self.optimizer = AdoptAtan2(
                self.world_model.parameters(),
                lr=self.lr,
                betas=self.betas
            )
            
            if self.accelerator:
                self.world_model, self.optimizer = self.accelerator.prepare(
                    self.world_model, self.optimizer
                )
    
    def get_fractal_info(self) -> Dict[str, Any]:
        """Get information about the fractal configuration."""
        return {
            'num_fractal_levels': self.num_fractal_levels,
            'fractal_embed_dim': self.fractal_embed_dim,
            'fractal_heads': self.fractal_heads,
            'fractal_share_weights': self.fractal_share_weights,
            'fractal_use_hypernetwork': self.fractal_use_hypernetwork,
            'fractal_global_state_dim': self.fractal_global_state_dim
        }
    
    def analyze_fractal_representations(
        self, 
        state: Tensor,
        actions: Optional[Tensor] = None,
        rewards: Optional[Tensor] = None
    ) -> Dict[str, Any]:
        """
        Analyze the fractal representations for debugging and interpretability.
        
        Returns detailed information about how information flows through
        different fractal levels.
        """
        with torch.no_grad():
            # Get detailed fractal processing info
            raw_actions, values, state_pred, dones, cache = self.world_model(
                state, actions=actions, rewards=rewards
            )
            
            fractal_levels = cache.get('fractal_levels', [])
            level_features = cache.get('level_features', [])
            
            analysis = {
                'num_levels_processed': len(fractal_levels),
                'level_shapes': [level.shape for level in fractal_levels],
                'level_feature_norms': [torch.norm(feat).item() for feat in level_features],
                'raw_actions_shape': raw_actions.shape,
                'values_shape': values.shape,
                'representation_diversity': self._compute_representation_diversity(fractal_levels),
                'inter_level_similarity': self._compute_inter_level_similarity(level_features)
            }
            
            return analysis
    
    def _compute_representation_diversity(self, fractal_levels: List[Tensor]) -> List[float]:
        """Compute diversity metrics for each fractal level."""
        diversities = []
        for level in fractal_levels:
            # Compute pairwise cosine similarities within the level
            level_flat = level.view(level.shape[0], -1)  # [batch, features]
            normalized = torch.nn.functional.normalize(level_flat, dim=1)
            similarity_matrix = torch.mm(normalized, normalized.t())
            
            # Diversity is 1 - mean similarity (excluding diagonal)
            mask = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
            off_diagonal = similarity_matrix[~mask.bool()]
            diversity = 1.0 - off_diagonal.mean().item()
            diversities.append(diversity)
        
        return diversities
    
    def _compute_inter_level_similarity(self, level_features: List[Tensor]) -> List[float]:
        """Compute similarity between adjacent fractal levels."""
        if len(level_features) < 2:
            return []
        
        similarities = []
        for i in range(len(level_features) - 1):
            feat1 = torch.nn.functional.normalize(level_features[i], dim=1)
            feat2 = torch.nn.functional.normalize(level_features[i + 1], dim=1)
            
            # Compute cosine similarity between level features
            similarity = torch.nn.functional.cosine_similarity(feat1, feat2, dim=1)
            similarities.append(similarity.mean().item())
        
        return similarities
    
    def save_fractal_config(self, path: Optional[str] = None):
        """Save fractal-specific configuration along with model state."""
        save_path = path or self.save_path
        
        state_dict = {
            'world_model': self.world_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'fractal_config': self.get_fractal_info(),
            'training_step': getattr(self, 'step', 0),
            'agent_config': {
                'state_dim': self.state_dim,
                'num_actions': self.num_actions,
                'reward_range': self.reward_range,
                'continuous_actions': self.continuous_actions,
                'evolutionary': self.evolutionary
            }
        }
        
        if self.evolutionary:
            state_dict['gene_pool'] = self.gene_pool.state_dict()
        
        torch.save(state_dict, save_path)
        print(f"Fractal agent saved to {save_path}")
    
    def load_fractal_config(self, path: Optional[str] = None):
        """Load fractal agent from saved state."""
        load_path = path or self.save_path
        
        if not torch.cuda.is_available():
            state_dict = torch.load(load_path, map_location='cpu')
        else:
            state_dict = torch.load(load_path)
        
        # Load world model
        self.world_model.load_state_dict(state_dict['world_model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        
        # Load fractal config for verification
        saved_config = state_dict.get('fractal_config', {})
        current_config = self.get_fractal_info()
        
        # Warn if configurations don't match
        for key, value in saved_config.items():
            if key in current_config and current_config[key] != value:
                print(f"Warning: {key} mismatch. Saved: {value}, Current: {current_config[key]}")
        
        if self.evolutionary and 'gene_pool' in state_dict:
            self.gene_pool.load_state_dict(state_dict['gene_pool'])
        
        print(f"Fractal agent loaded from {load_path}")
        return state_dict.get('training_step', 0)


class FractalLearner:
    """
    A high-level interface for training fractal RL agents, similar to the
    original Learner class but with fractal-specific enhancements.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        reward_range: Tuple[float, float],
        continuous_actions: bool = False,
        squash_continuous: bool = True,
        continuous_actions_clamp: Optional[Tuple[float, float]] = None,
        evolutionary: bool = False,
        evolve_every: int = 10,
        evolve_after_step: int = 20,
        latent_gene_pool: Optional[dict] = None,
        max_timesteps: int = 500,
        batch_size: int = 8,
        num_episodes_per_update: int = 64,
        lr: float = 0.0008,
        betas: Tuple[float, float] = (0.9, 0.99),
        lam: float = 0.95,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        value_clip: float = 0.4,
        beta_s: float = .01,
        regen_reg_rate: float = 1e-4,
        cautious_factor: float = 0.1,
        epochs: int = 4,
        ema_decay: float = 0.9,
        save_every: int = 100,
        frac_actor_critic_head_gradient: float = 0.5,
        # Fractal-specific parameters
        num_fractal_levels: int = 4,
        fractal_embed_dim: int = 256,
        fractal_heads: int = 8,
        fractal_share_weights: bool = False,
        fractal_use_hypernetwork: bool = False,
        analyze_fractal_every: int = 50,  # How often to analyze fractal representations
        accelerate_kwargs: dict = dict(),
        agent_kwargs: dict = dict()
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.reward_range = reward_range
        self.continuous_actions = continuous_actions
        self.max_timesteps = max_timesteps
        self.num_episodes_per_update = num_episodes_per_update
        self.save_every = save_every
        self.analyze_fractal_every = analyze_fractal_every
        
        # Setup accelerator
        from accelerate import Accelerator
        self.accelerator = Accelerator(**accelerate_kwargs)
        
        # Create fractal agent
        self.agent = FractalAgent(
            state_dim=state_dim,
            num_actions=num_actions,
            reward_range=reward_range,
            max_timesteps=max_timesteps,
            batch_size=batch_size,
            lr=lr,
            betas=betas,
            lam=lam,
            gamma=gamma,
            beta_s=beta_s,
            regen_reg_rate=regen_reg_rate,
            cautious_factor=cautious_factor,
            eps_clip=eps_clip,
            value_clip=value_clip,
            ema_decay=ema_decay,
            continuous_actions=continuous_actions,
            squash_continuous=squash_continuous,
            evolutionary=evolutionary,
            evolve_every=evolve_every,
            evolve_after_step=evolve_after_step,
            latent_gene_pool=latent_gene_pool or dict(),
            epochs=epochs,
            frac_actor_critic_head_gradient=frac_actor_critic_head_gradient,
            num_fractal_levels=num_fractal_levels,
            fractal_embed_dim=fractal_embed_dim,
            fractal_heads=fractal_heads,
            fractal_share_weights=fractal_share_weights,
            fractal_use_hypernetwork=fractal_use_hypernetwork,
            accelerator=self.accelerator,
            **agent_kwargs
        )
        
        self.fractal_analysis_history = []
    
    @property
    def device(self):
        return self.accelerator.device
    
    def train(
        self,
        env,
        num_learning_updates: int,
        seed: Optional[int] = None,
        max_timesteps: Optional[int] = None
    ):
        """
        Train the fractal agent with enhanced monitoring and analysis.
        """
        from x_transformers_rl.distributed import maybe_sync_seed
        import random
        from tqdm import tqdm
        
        maybe_sync_seed(seed)
        if exists(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        max_timesteps = default(max_timesteps, self.max_timesteps)
        
        for update_step in tqdm(range(num_learning_updates), desc='Training Fractal Agent'):
            
            # Collect episodes
            episodes = []
            episode_lens = []
            gene_ids = []
            
            for episode_idx in range(self.num_episodes_per_update):
                memories, episode_len, gene_id = self._collect_episode(
                    env, max_timesteps, episode_idx
                )
                episodes.append(memories)
                episode_lens.append(episode_len)
                gene_ids.append(gene_id)
            
            # Learn from collected episodes
            metrics = self.agent.learn(episodes, episode_lens, gene_ids)
            
            # Fractal analysis
            if update_step % self.analyze_fractal_every == 0:
                self._analyze_fractal_representations(episodes[0])
            
            # Save periodically
            if update_step % self.save_every == 0:
                self.agent.save_fractal_config()
            
            # Log metrics
            if update_step % 10 == 0:
                self._log_metrics(update_step, metrics)
    
    def _collect_episode(self, env, max_timesteps: int, episode_idx: int):
        """Collect a single episode using the fractal agent."""
        memories = []
        state = env.reset()
        
        prev_action = None
        prev_reward = torch.tensor(0.)
        
        gene_id = 0
        if self.agent.evolutionary:
            gene_id = self.agent.gene_pool.sample()
        
        for timestep in range(max_timesteps):
            state_tensor = from_numpy(state).unsqueeze(0).to(self.device)
            
            # Get action from fractal agent
            action, value, _, latent_gene = self.agent(
                state_tensor,
                reward=prev_reward.unsqueeze(0).to(self.device) if exists(prev_reward) else None,
                latent_gene_id=gene_id
            )
            
            # Execute action in environment
            next_state, reward, done, info = env.step(action.cpu().numpy())
            
            # Store memory
            memory = create_memory(
                state_tensor,
                action,
                torch.log(torch.tensor(1.0)),  # Placeholder log prob
                torch.tensor(reward),
                torch.tensor(done),
                value
            )
            memories.append(memory)
            
            if done:
                break
            
            state = next_state
            prev_action = action
            prev_reward = torch.tensor(reward)
        
        return memories, len(memories), gene_id
    
    def _analyze_fractal_representations(self, episode_memories: List[Memory]):
        """Analyze fractal representations from an episode."""
        if not episode_memories:
            return
        
        # Take a sample of states from the episode
        sample_indices = torch.linspace(0, len(episode_memories) - 1, 
                                      min(10, len(episode_memories))).long()
        
        sample_states = torch.stack([episode_memories[i].state for i in sample_indices])
        sample_actions = torch.stack([episode_memories[i].action for i in sample_indices])
        
        analysis = self.agent.analyze_fractal_representations(
            sample_states.squeeze(1),  # Remove batch dim that was added during collection
            actions=sample_actions.squeeze(1)
        )
        
        self.fractal_analysis_history.append(analysis)
        
        # Print key insights
        print(f"\nFractal Analysis:")
        print(f"  Levels processed: {analysis['num_levels_processed']}")
        print(f"  Representation diversity: {[f'{d:.3f}' for d in analysis['representation_diversity']]}")
        print(f"  Inter-level similarity: {[f'{s:.3f}' for s in analysis['inter_level_similarity']]}")
    
    def _log_metrics(self, step: int, metrics: dict):
        """Log training metrics."""
        print(f"\nStep {step}:")
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.item()
            print(f"  {key}: {value:.6f}")


# Example usage and demo
def create_simple_fractal_env():
    """Create a simple environment for testing fractal agents."""
    import gym
    
    # For demo purposes, we'll use a simple environment
    # In practice, you'd use more complex environments
    return gym.make('CartPole-v1')


def demo_fractal_agent():
    """Demonstrate the fractal agent on a simple environment."""
    print("Creating fractal RL agent demo...")
    
    # Environment setup
    env = create_simple_fractal_env()
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    # Create fractal learner
    learner = FractalLearner(
        state_dim=state_dim,
        num_actions=num_actions,
        reward_range=(-1.0, 1.0),
        num_fractal_levels=3,
        fractal_embed_dim=128,
        fractal_heads=4,
        max_timesteps=200,
        num_episodes_per_update=4,
        analyze_fractal_every=5
    )
    
    print(f"Fractal agent configuration: {learner.agent.get_fractal_info()}")
    
    # Train for a few updates
    try:
        learner.train(env, num_learning_updates=20, seed=42)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training error: {e}")
        print("This is expected in a demo without proper environment setup")
    
    env.close()


if __name__ == "__main__":
    demo_fractal_agent() 