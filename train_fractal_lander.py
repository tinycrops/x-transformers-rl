#!/usr/bin/env python3
"""
Fractal Agent Training on LunarLander-v2

This script demonstrates training the Fractal Reinforcement Learning Agent (FRALA)
on the LunarLander-v2 environment. It shows how the fractal architecture processes
information at multiple scales and maintains a global state across levels.

Usage:
    python train_fractal_lander.py
    
Key Features:
- Multi-level fractal processing inspired by HLIP's hierarchical attention
- Global state management across fractal levels
- Detailed analysis and visualization of fractal representations
- Comparison with standard transformer-based agents
"""

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import json

from x_transformers_rl.fractal_agent import FractalAgent, FractalLearner
from x_transformers_rl.x_transformers_rl import Agent, Learner


def create_fractal_config(
    difficulty: str = "medium",
    use_hierarchical_levels: bool = True
) -> Dict[str, Any]:
    """
    Create fractal configuration for different difficulty levels.
    
    Args:
        difficulty: "easy", "medium", "hard", or "extreme"
        use_hierarchical_levels: Whether to use hierarchical processing levels
    """
    
    base_config = {
        "state_dim": 8,  # LunarLander observation space
        "num_actions": 4,  # LunarLander action space  
        "reward_range": (-300.0, 300.0),  # Approximate LunarLander reward range
        "continuous_actions": False,
        "max_timesteps": 1000,
        "batch_size": 32,
        "num_episodes_per_update": 16,
        "lr": 0.0003,
        "gamma": 0.99,
        "lam": 0.95,
        "eps_clip": 0.2,
        "value_clip": 0.4,
        "epochs": 4,
        "save_every": 25,
        "analyze_fractal_every": 10
    }
    
    # Fractal-specific configurations based on difficulty
    fractal_configs = {
        "easy": {
            "num_fractal_levels": 2,
            "fractal_embed_dim": 128,
            "fractal_heads": 4,
            "fractal_dim_head": 32,
            "fractal_share_weights": True,
            "fractal_use_hypernetwork": False
        },
        "medium": {
            "num_fractal_levels": 3,
            "fractal_embed_dim": 256,
            "fractal_heads": 8,
            "fractal_dim_head": 32,
            "fractal_share_weights": False,
            "fractal_use_hypernetwork": False
        },
        "hard": {
            "num_fractal_levels": 4,
            "fractal_embed_dim": 512,
            "fractal_heads": 12,
            "fractal_dim_head": 64,
            "fractal_share_weights": False,
            "fractal_use_hypernetwork": True
        },
        "extreme": {
            "num_fractal_levels": 6,
            "fractal_embed_dim": 768,
            "fractal_heads": 16,
            "fractal_dim_head": 48,
            "fractal_share_weights": False,
            "fractal_use_hypernetwork": True,
            "fractal_global_state_dim": 1024
        }
    }
    
    base_config.update(fractal_configs[difficulty])
    return base_config


class FractalExperiment:
    """
    Experimental framework for training and analyzing fractal agents.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        exp_name: str = "fractal_lander",
        save_dir: str = "./fractal_experiments"
    ):
        self.config = config
        self.exp_name = exp_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create environment
        self.env = gym.make('LunarLander-v3')
        
        # Separate config into learner and agent parameters
        agent_only_params = {
            'fractal_dim_head', 'fractal_ff_mult', 'fractal_global_state_dim'
        }
        
        learner_config = {k: v for k, v in config.items() if k not in agent_only_params}
        agent_kwargs = {k: v for k, v in config.items() if k in agent_only_params}
        
        # Create fractal learner
        self.learner = FractalLearner(**learner_config, agent_kwargs=agent_kwargs)
        
        # Tracking
        self.training_history = []
        self.fractal_analysis_history = []
        self.episode_rewards = []
        
        print(f"Fractal Experiment: {exp_name}")
        print(f"Configuration: {config}")
        print(f"Fractal Info: {self.learner.agent.get_fractal_info()}")
    
    def train(
        self, 
        num_updates: int = 500,
        seed: Optional[int] = 42,
        verbose: bool = True
    ):
        """Train the fractal agent with detailed logging."""
        
        print(f"\nStarting training for {num_updates} updates...")
        
        # Custom training loop for better control and analysis
        for update in range(num_updates):
            
            # Collect episodes
            episode_rewards_batch = []
            episode_lengths_batch = []
            
            for episode_idx in range(self.config['num_episodes_per_update']):
                episode_reward, episode_length = self._run_episode(seed + update * 1000 + episode_idx)
                episode_rewards_batch.append(episode_reward)
                episode_lengths_batch.append(episode_length)
            
            # Store metrics
            avg_reward = np.mean(episode_rewards_batch)
            avg_length = np.mean(episode_lengths_batch)
            
            self.episode_rewards.extend(episode_rewards_batch)
            
            metrics = {
                'update': update,
                'avg_reward': avg_reward,
                'avg_length': avg_length,
                'max_reward': np.max(episode_rewards_batch),
                'min_reward': np.min(episode_rewards_batch),
                'reward_std': np.std(episode_rewards_batch)
            }
            
            self.training_history.append(metrics)
            
            # Fractal analysis
            if update % self.config.get('analyze_fractal_every', 10) == 0:
                analysis = self._analyze_current_policy()
                self.fractal_analysis_history.append({
                    'update': update,
                    **analysis
                })
                
                if verbose:
                    self._print_fractal_analysis(update, analysis)
            
            # Logging
            if update % 10 == 0 and verbose:
                print(f"Update {update:3d} | "
                      f"Reward: {avg_reward:8.2f} ± {np.std(episode_rewards_batch):6.2f} | "
                      f"Length: {avg_length:6.1f} | "
                      f"Success Rate: {self._compute_success_rate(episode_rewards_batch):5.1%}")
            
            # Save periodically
            if update % self.config.get('save_every', 25) == 0:
                self._save_checkpoint(update)
        
        print(f"\nTraining completed! Final average reward: {avg_reward:.2f}")
        self._save_final_results()
    
    def _run_episode(self, seed: int) -> tuple[float, int]:
        """Run a single episode and return reward and length."""
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        state, _ = self.env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        
        with torch.no_grad():
            for step in range(self.config['max_timesteps']):
                
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state)
                
                # Get action from fractal agent
                result = self.learner.agent(state_tensor)
                action = result[0] if isinstance(result, tuple) else result
                action = action.squeeze().argmax().item()
                
                # Take environment step
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
                
                state = next_state
        
        return total_reward, steps
    
    def _analyze_current_policy(self) -> Dict[str, Any]:
        """Analyze the current policy's fractal representations."""
        
        # Run a sample episode and collect states
        state, _ = self.env.reset(seed=12345)
        states = []
        actions = []
        
        with torch.no_grad():
            for _ in range(min(50, self.config['max_timesteps'])):
                state_tensor = torch.FloatTensor(state)
                states.append(state_tensor.unsqueeze(0))  # Add batch dim for analysis
                
                result = self.learner.agent(state_tensor)
                action = result[0] if isinstance(result, tuple) else result
                action_idx = action.squeeze().argmax().item()
                actions.append(torch.tensor([action_idx]))
                
                next_state, _, terminated, truncated, _ = self.env.step(action_idx)
                
                if terminated or truncated:
                    break
                
                state = next_state
        
        if not states:
            return {}
        
        # Analyze fractal representations
        states_batch = torch.cat(states[:10])  # Analyze first 10 states
        actions_batch = torch.cat(actions[:10]) if actions else None
        
        # Ensure tensors are on the correct device
        if hasattr(self.learner.agent, 'device'):
            states_batch = states_batch.to(self.learner.agent.device)
            if actions_batch is not None:
                actions_batch = actions_batch.to(self.learner.agent.device)
        
        analysis = self.learner.agent.analyze_fractal_representations(
            states_batch, actions=actions_batch
        )
        
        return analysis
    
    def _compute_success_rate(self, rewards: List[float]) -> float:
        """Compute success rate (rewards > 200 are generally considered successful)."""
        return sum(1 for r in rewards if r > 200) / len(rewards)
    
    def _print_fractal_analysis(self, update: int, analysis: Dict[str, Any]):
        """Print fractal analysis in a readable format."""
        print(f"\n--- Fractal Analysis (Update {update}) ---")
        print(f"Levels processed: {analysis.get('num_levels_processed', 'N/A')}")
        
        diversity = analysis.get('representation_diversity', [])
        if diversity:
            print(f"Representation diversity: {[f'{d:.3f}' for d in diversity]}")
        
        similarity = analysis.get('inter_level_similarity', [])
        if similarity:
            print(f"Inter-level similarity: {[f'{s:.3f}' for s in similarity]}")
        
        feature_norms = analysis.get('level_feature_norms', [])
        if feature_norms:
            print(f"Feature magnitudes: {[f'{n:.2f}' for n in feature_norms]}")
    
    def _save_checkpoint(self, update: int):
        """Save model checkpoint and training data."""
        checkpoint_dir = self.save_dir / f"{self.exp_name}_update_{update}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        self.learner.agent.save_fractal_config(
            str(checkpoint_dir / "fractal_agent.pt")
        )
        
        # Save training history
        with open(checkpoint_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save fractal analysis
        with open(checkpoint_dir / "fractal_analysis.json", 'w') as f:
            json.dump(self.fractal_analysis_history, f, indent=2)
    
    def _save_final_results(self):
        """Save final experimental results and create visualizations."""
        final_dir = self.save_dir / f"{self.exp_name}_final"
        final_dir.mkdir(exist_ok=True)
        
        # Save final model
        self.learner.agent.save_fractal_config(
            str(final_dir / "final_fractal_agent.pt")
        )
        
        # Save all data
        results = {
            'config': self.config,
            'training_history': self.training_history,
            'fractal_analysis_history': self.fractal_analysis_history,
            'episode_rewards': self.episode_rewards,
            'final_info': self.learner.agent.get_fractal_info()
        }
        
        with open(final_dir / "experiment_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        self._create_visualizations(final_dir)
        
        print(f"Results saved to: {final_dir}")
    
    def _create_visualizations(self, save_dir: Path):
        """Create training and fractal analysis visualizations."""
        
        # Training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward over time
        rewards = [h['avg_reward'] for h in self.training_history]
        axes[0, 0].plot(rewards)
        axes[0, 0].set_title('Average Reward over Training')
        axes[0, 0].set_xlabel('Update')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True)
        
        # Episode length over time
        lengths = [h['avg_length'] for h in self.training_history]
        axes[0, 1].plot(lengths)
        axes[0, 1].set_title('Average Episode Length')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Average Length')
        axes[0, 1].grid(True)
        
        # Success rate
        success_rates = []
        window_size = 50
        for i in range(len(self.episode_rewards)):
            start_idx = max(0, i - window_size)
            window_rewards = self.episode_rewards[start_idx:i+1]
            success_rate = self._compute_success_rate(window_rewards)
            success_rates.append(success_rate)
        
        axes[1, 0].plot(success_rates)
        axes[1, 0].set_title(f'Success Rate (rolling window={window_size})')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].grid(True)
        
        # Fractal analysis over time
        if self.fractal_analysis_history:
            updates = [h['update'] for h in self.fractal_analysis_history]
            
            # Plot diversity of first level if available
            diversities = []
            for h in self.fractal_analysis_history:
                div = h.get('representation_diversity', [])
                diversities.append(div[0] if div else 0)
            
            axes[1, 1].plot(updates, diversities)
            axes[1, 1].set_title('Representation Diversity (Level 0)')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Diversity')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Fractal-specific visualizations
        if self.fractal_analysis_history:
            self._create_fractal_visualizations(save_dir)
    
    def _create_fractal_visualizations(self, save_dir: Path):
        """Create detailed fractal analysis visualizations."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        updates = [h['update'] for h in self.fractal_analysis_history]
        
        # Representation diversity across levels
        num_levels = self.config['num_fractal_levels']
        for level in range(num_levels):
            diversities = []
            for h in self.fractal_analysis_history:
                div = h.get('representation_diversity', [])
                diversities.append(div[level] if len(div) > level else 0)
            axes[0, 0].plot(updates, diversities, label=f'Level {level}')
        
        axes[0, 0].set_title('Representation Diversity by Fractal Level')
        axes[0, 0].set_xlabel('Update')
        axes[0, 0].set_ylabel('Diversity')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Inter-level similarity
        for i in range(num_levels - 1):
            similarities = []
            for h in self.fractal_analysis_history:
                sim = h.get('inter_level_similarity', [])
                similarities.append(sim[i] if len(sim) > i else 0)
            axes[0, 1].plot(updates, similarities, label=f'Level {i}→{i+1}')
        
        axes[0, 1].set_title('Inter-Level Similarity')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Similarity')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Feature magnitude by level
        for level in range(num_levels):
            magnitudes = []
            for h in self.fractal_analysis_history:
                mags = h.get('level_feature_norms', [])
                magnitudes.append(mags[level] if len(mags) > level else 0)
            axes[1, 0].plot(updates, magnitudes, label=f'Level {level}')
        
        axes[1, 0].set_title('Feature Magnitude by Fractal Level')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('L2 Norm')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Heatmap of final inter-level similarities
        if self.fractal_analysis_history:
            final_analysis = self.fractal_analysis_history[-1]
            similarities = final_analysis.get('inter_level_similarity', [])
            
            if similarities:
                # Create similarity matrix
                sim_matrix = np.zeros((num_levels, num_levels))
                for i, sim in enumerate(similarities):
                    sim_matrix[i, i+1] = sim
                    sim_matrix[i+1, i] = sim  # Make symmetric
                
                np.fill_diagonal(sim_matrix, 1.0)  # Self-similarity = 1
                
                sns.heatmap(sim_matrix, annot=True, cmap='viridis', 
                           ax=axes[1, 1], square=True)
                axes[1, 1].set_title('Final Inter-Level Similarity Matrix')
                axes[1, 1].set_xlabel('Fractal Level')
                axes[1, 1].set_ylabel('Fractal Level')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'fractal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def run_fractal_experiment(
    difficulty: str = "medium",
    num_updates: int = 500,
    exp_name: Optional[str] = None,
    seed: int = 42
):
    """Run a complete fractal experiment."""
    
    if exp_name is None:
        exp_name = f"fractal_lander_{difficulty}"
    
    # Create configuration
    config = create_fractal_config(difficulty)
    
    # Create and run experiment
    experiment = FractalExperiment(config, exp_name)
    experiment.train(num_updates=num_updates, seed=seed)
    
    return experiment


def compare_fractal_vs_standard(num_updates: int = 200):
    """Compare fractal agent with standard transformer agent."""
    
    print("Running comparison: Fractal vs Standard Agent")
    
    # Run fractal experiment
    fractal_config = create_fractal_config("medium")
    fractal_exp = FractalExperiment(fractal_config, "fractal_comparison")
    
    print("\n=== Training Fractal Agent ===")
    fractal_exp.train(num_updates=num_updates, verbose=False)
    fractal_rewards = fractal_exp.episode_rewards[-50:]  # Last 50 episodes
    
    print(f"Fractal Agent - Final avg reward: {np.mean(fractal_rewards):.2f} ± {np.std(fractal_rewards):.2f}")
    
    # Note: Standard agent comparison would require implementing a standard transformer agent
    # For now, we'll just report the fractal agent results
    
    return {
        'fractal_rewards': fractal_rewards,
        'fractal_analysis': fractal_exp.fractal_analysis_history
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Fractal Agent on LunarLander")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "extreme"], 
                       default="medium", help="Fractal configuration difficulty")
    parser.add_argument("--updates", type=int, default=300, 
                       help="Number of training updates")
    parser.add_argument("--exp-name", type=str, default=None,
                       help="Experiment name (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compare", action="store_true",
                       help="Run comparison with standard agent")
    
    args = parser.parse_args()
    
    if args.compare:
        results = compare_fractal_vs_standard(args.updates)
        print(f"Comparison completed. Results: {len(results['fractal_rewards'])} episodes analyzed.")
    else:
        experiment = run_fractal_experiment(
            difficulty=args.difficulty,
            num_updates=args.updates,
            exp_name=args.exp_name,
            seed=args.seed
        )
        
        print(f"\nExperiment completed successfully!")
        print(f"Final configuration: {experiment.learner.agent.get_fractal_info()}")
        
        if experiment.episode_rewards:
            final_rewards = experiment.episode_rewards[-50:]
            print(f"Final performance: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
            print(f"Success rate: {experiment._compute_success_rate(final_rewards):.1%}") 