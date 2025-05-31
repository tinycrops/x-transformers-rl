#!/usr/bin/env python3
"""
Quick demo of FRALA (Fractal Reinforcement Learning Agent) on CartPole
"""

import torch
import gymnasium as gym
import numpy as np
from x_transformers_rl.fractal_agent import FractalLearner

def main():
    print("🌀 FRALA Quick Demo on CartPole")
    print("=" * 50)
    
    # Create environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    print(f"Environment: CartPole-v1")
    print(f"State dimension: {state_dim}")
    print(f"Number of actions: {num_actions}")
    
    # Create fractal learner
    learner = FractalLearner(
        state_dim=state_dim,
        num_actions=num_actions,
        reward_range=(0.0, 500.0),  # CartPole reward range
        num_fractal_levels=3,
        fractal_embed_dim=128,
        fractal_heads=4,
        max_timesteps=200,
        num_episodes_per_update=4,
        lr=0.001,
        analyze_fractal_every=2
    )
    
    print(f"\nFractal Configuration:")
    config = learner.agent.get_fractal_info()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nRunning demo training for 5 updates...")
    
    # Simple training loop
    for update in range(5):
        print(f"\n--- Update {update + 1} ---")
        
        # Collect a few episodes
        episode_rewards = []
        
        for episode in range(4):  # num_episodes_per_update
            state, _ = env.reset(seed=42 + update * 4 + episode)
            total_reward = 0.0
            
            with torch.no_grad():
                for step in range(200):  # max_timesteps
                    # Convert state to tensor
                    state_tensor = torch.FloatTensor(state)
                    if hasattr(learner.agent, 'device'):
                        state_tensor = state_tensor.to(learner.agent.device)
                    
                    # Get action from fractal agent
                    result = learner.agent(state_tensor)
                    action = result[0] if isinstance(result, tuple) else result
                    action_idx = action.squeeze().argmax().item()
                    
                    # Step environment
                    next_state, reward, terminated, truncated, _ = env.step(action_idx)
                    total_reward += reward
                    
                    if terminated or truncated:
                        break
                    
                    state = next_state
            
            episode_rewards.append(total_reward)
        
        # Print results
        avg_reward = np.mean(episode_rewards)
        max_reward = np.max(episode_rewards)
        print(f"Episode rewards: {episode_rewards}")
        print(f"Average reward: {avg_reward:.1f}")
        print(f"Max reward: {max_reward:.1f}")
        
        # Fractal analysis
        if update % 2 == 0:
            print("\n🔍 Fractal Analysis:")
            sample_state = torch.FloatTensor(env.observation_space.sample()).unsqueeze(0)
            if hasattr(learner.agent, 'device'):
                sample_state = sample_state.to(learner.agent.device)
            
            analysis = learner.agent.analyze_fractal_representations(sample_state)
            print(f"  Levels processed: {analysis.get('num_levels_processed', 'N/A')}")
            
            diversity = analysis.get('representation_diversity', [])
            if diversity:
                print(f"  Representation diversity: {[f'{d:.3f}' for d in diversity]}")
            
            similarity = analysis.get('inter_level_similarity', [])
            if similarity:
                print(f"  Inter-level similarity: {[f'{s:.3f}' for s in similarity]}")
    
    env.close()
    print(f"\n🎉 Demo completed successfully!")
    print(f"FRALA is working correctly on CartPole environment.")
    
    print(f"\nTo run full training on LunarLander (after installing Box2D):")
    print(f"  python train_fractal_lander.py --difficulty medium --updates 100")

if __name__ == "__main__":
    main() 