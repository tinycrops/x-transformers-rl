# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
```bash
pytest
```

### Installation 
For development:
```bash
pip install -r requirements.txt
```

For package installation:
```bash
pip install x-transformers-rl
```

### Running Examples
```bash
python train_lander.py
```

## Architecture

This is a reinforcement learning library that implements transformers for RL using the x-transformers library. The core architecture combines:

### Main Components

1. **WorldModelActorCritic** (`x_transformers_rl/x_transformers_rl.py:281-560`): The central model that combines:
   - Transformer-based world model for predicting next states and rewards
   - Actor head for action prediction (discrete or continuous)
   - Critic head using HL-Gauss loss for value prediction
   - Evolutionary latent gene integration when enabled

2. **Agent** (`x_transformers_rl/x_transformers_rl.py:644-1066`): PPO-based agent that:
   - Uses EMA (Exponential Moving Average) for model updates
   - Implements RSNorm for state normalization
   - Handles both discrete and continuous action spaces
   - Integrates with evolutionary gene pool when enabled

3. **Learner** (`x_transformers_rl/x_transformers_rl.py:1069-1381`): Main training wrapper that:
   - Orchestrates environment interaction
   - Manages parallel episode collection
   - Handles distributed training via Accelerate
   - Coordinates evolutionary algorithms when enabled

4. **LatentGenePool** (`x_transformers_rl/evolution.py:28-185`): Evolutionary component implementing:
   - Tournament selection
   - Crossover and mutation operations
   - Multi-island evolution with migration
   - L2-normalized latent gene vectors

### Key Features

- **Evolutionary Policy Optimization**: Optional evolutionary algorithm that evolves latent genes to improve policy performance
- **World Model Learning**: Autoregressive prediction of next states and rewards
- **Distributed Training**: Built-in support for multi-GPU training via Accelerate
- **Action Space Flexibility**: Supports both discrete and continuous actions with optional squashing
- **State Normalization**: RSNorm for online state and reward normalization

### Loss Components

The training combines three loss types:
1. **Actor Loss**: PPO clipped surrogate objective with entropy regularization
2. **Critic Loss**: Clipped value loss using HL-Gauss distribution
3. **World Model Loss**: Autoregressive prediction loss for states/rewards + done prediction

### Memory Structure

Episodes are stored as `Memory` namedtuples containing:
- `state`, `action`, `action_log_prob`, `reward`, `is_boundary`, `value`

GAE (Generalized Advantage Estimation) is used for advantage calculation with associative scan for efficient computation.