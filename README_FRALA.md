# FRALA: Fractal Reinforcement Learning Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FRALA** (Fractal Reinforcement Learning Agent) is a novel RL architecture that processes information and makes decisions based on hierarchical, self-similar internal representations. This implementation combines the hierarchical attention mechanisms from HLIP (Hierarchical Language-Image Pre-training) with the reinforcement learning framework from x-transformers-rl.

## 🎯 Core Concept

The fractal agent is inspired by the concept where patterns repeat at different scales, and a "global" aspect of the agent is consistent across these scales. Like in the original fractal game concept, actions affecting one "copy" (level) affect the global entity through a shared state.

### Key Principles

1. **Multiple Levels/Scales**: The agent processes information at different "zoom levels" or scales
2. **Self-Similarity**: The structure or processing at one level is similar to others 
3. **Shared Global State**: A central latent state ("The Soul") that maintains consistency across all scales
4. **Inter-Level Communication**: Information flows between levels through upscaling and downscaling operations

## 🏗️ Architecture

```
Input State → Initial Embedding
     ↓
Level 0 (Coarsest): FractalBlock_0 → Update GlobalState
     ↓ ↕ (upscale/downscale)
Level 1: FractalBlock_1 → Update GlobalState  
     ↓ ↕ 
Level N-1 (Finest): FractalBlock_{N-1} → Update GlobalState
     ↓
Aggregation: Combine all level features + GlobalState
     ↓
Actor-Critic Heads → Actions & Values
```

### Components

- **FractalEncoder**: Core multi-level processing with self-similar blocks
- **FractalProcessingBlock**: Self-similar transformer blocks with global attention
- **Global State Management**: Shared latent state across all fractal levels
- **Inter-Level Transformations**: HLIP-inspired upscaling/downscaling operations
- **FractalWorldModelActorCritic**: Integration with x-transformers-rl framework

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd x-transformers-rl

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for fractal implementation
pip install einops einx matplotlib seaborn gymnasium[classic_control]
```

### Basic Usage

```python
from x_transformers_rl.fractal_agent import FractalAgent, FractalLearner

# Create a fractal learner for LunarLander
learner = FractalLearner(
    state_dim=8,
    num_actions=4,
    reward_range=(-300.0, 300.0),
    num_fractal_levels=3,
    fractal_embed_dim=256,
    fractal_heads=8
)

# Train the agent
import gymnasium as gym
env = gym.make('LunarLander-v2')
learner.train(env, num_learning_updates=500)
```

### Training on LunarLander

```bash
# Basic training
python train_fractal_lander.py

# With different difficulty levels
python train_fractal_lander.py --difficulty medium --updates 500

# Available difficulties: easy, medium, hard, extreme
python train_fractal_lander.py --difficulty extreme --updates 1000 --exp-name my_experiment
```

## 📊 Fractal Analysis

The implementation provides detailed analysis of fractal representations:

### Metrics Tracked

- **Representation Diversity**: How different the features are within each level
- **Inter-Level Similarity**: How similar adjacent fractal levels are
- **Feature Magnitudes**: The L2 norm of features at each level
- **Global State Evolution**: How the shared state changes over time

### Example Analysis Output

```
--- Fractal Analysis (Update 50) ---
Levels processed: 3
Representation diversity: ['0.742', '0.681', '0.598']
Inter-level similarity: ['0.823', '0.756']
Feature magnitudes: ['12.34', '8.91', '6.78']
```

## 🎮 Difficulty Configurations

### Easy (2 levels)
- Fast training, simple hierarchical processing
- Shared weights across levels
- Good for quick experiments

### Medium (3 levels) 
- Balanced complexity and performance
- Separate blocks per level
- Recommended starting point

### Hard (4 levels)
- More complex hierarchical processing
- Uses hypernetwork for level-specific weights
- Better for complex environments

### Extreme (6 levels)
- Maximum hierarchical depth
- Large embedding dimensions
- For research and complex tasks

## 🔬 Advanced Features

### Evolutionary Integration

```python
learner = FractalLearner(
    # ... other params ...
    evolutionary=True,
    evolve_every=10,
    latent_gene_pool={
        'dim': 128,
        'num_genes_per_island': 5,
        'num_selected': 3
    }
)
```

### Custom Fractal Configurations

```python
from x_transformers_rl.fractal_rl import FractalEncoder

# Create custom fractal encoder
encoder = FractalEncoder(
    input_dim=64,
    embed_dim=512,
    num_levels=4,
    heads=8,
    share_weights=False,
    use_hypernetwork=True,
    global_state_dim=256
)
```

### Analysis and Visualization

The training script automatically generates:
- Training curves (reward, episode length, success rate)
- Fractal analysis plots (diversity, similarity, magnitudes)
- Inter-level similarity heatmaps
- Checkpoints and experiment data

## 📈 Comparison with Standard Agents

To compare fractal vs standard transformer agents:

```bash
python train_fractal_lander.py --compare --updates 200
```

## 🛠️ Implementation Details

### HLIP Integration

The fractal architecture adapts HLIP's hierarchical attention mechanisms:

- `upscale_level()`: Aggregates fine features to coarse level (inspired by `_slice2scan`)
- `downscale_level()`: Distributes coarse features to fine level (inspired by `_study2slice`)
- Level-specific embeddings: Scale-aware positional embeddings
- Self-similar processing: Transformer blocks that can share weights or use hypernetworks

### x-transformers-rl Integration

- Compatible with existing `Agent` and `Learner` classes
- Supports evolutionary algorithms via `LatentGenePool`
- Maintains PPO training loop and loss functions
- Integrates with Accelerate for distributed training

## 📁 File Structure

```
x_transformers_rl/
├── fractal_rl.py          # Core fractal architecture
├── fractal_agent.py       # Agent and learner classes
├── x_transformers_rl.py   # Original RL framework
├── evolution.py           # Evolutionary algorithms
└── distributed.py         # Distributed training

train_fractal_lander.py    # Complete training script
README_FRALA.md           # This documentation
```

## 🧪 Experiments and Research

### Key Research Questions

1. **Scale Hierarchies**: How does performance scale with number of fractal levels?
2. **Weight Sharing**: When is weight sharing beneficial vs. level-specific parameters?
3. **Global State**: How does the global state evolve and what does it represent?
4. **Transfer Learning**: Do fractal representations transfer better across tasks?

### Experimental Framework

The `FractalExperiment` class provides:
- Systematic hyperparameter sweeps
- Detailed logging and analysis
- Automatic visualization generation
- Checkpoint management
- Reproducible experiments

## 🎯 Use Cases

### Ideal for:
- **Hierarchical Decision Making**: Tasks with natural scale hierarchies
- **Complex State Spaces**: Environments requiring multi-scale reasoning  
- **Long-Horizon Planning**: Tasks where global state persistence helps
- **Transfer Learning**: Leveraging self-similar structures across domains

### Examples:
- **Game Playing**: Strategy games with tactical and strategic levels
- **Robotics**: Multi-scale motor control and planning
- **Financial Trading**: Short-term and long-term market dynamics
- **Resource Management**: Local and global optimization problems

## 🔍 Debugging and Visualization

### Fractal Representation Analysis

```python
# Analyze current policy
analysis = agent.analyze_fractal_representations(states, actions)

# Check representation diversity
print(f"Level diversities: {analysis['representation_diversity']}")

# Examine inter-level communication
print(f"Level similarities: {analysis['inter_level_similarity']}")
```

### Common Issues

1. **Gradient Flow**: Check `frac_actor_critic_head_gradient` parameter
2. **Level Imbalance**: Monitor feature magnitudes across levels
3. **Global State Collapse**: Watch for diminishing global state updates
4. **Memory Usage**: Reduce `fractal_embed_dim` if experiencing OOM

## 🤝 Contributing

We welcome contributions! Areas of interest:

- **New Environments**: Testing on different RL environments
- **Architecture Variants**: Novel fractal processing patterns
- **Analysis Tools**: Better visualization and interpretation methods
- **Optimization**: Performance improvements and memory efficiency

## 📚 References

1. **HLIP Paper**: "Towards Scalable Language-Image Pre-training for 3D Medical Imaging" - Hierarchical attention mechanisms
2. **x-transformers-rl**: Foundation RL framework with transformer architectures
3. **Fractal Game Concept**: Original inspiration for multi-scale, self-similar processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HLIP Team**: For the hierarchical attention mechanisms that inspired this work
- **x-transformers-rl**: For the solid RL foundation and evolutionary algorithms
- **Fractal Gaming Community**: For the original concept that sparked this research direction

---

**Happy Fractal Learning!** 🌀🤖

For questions, issues, or discussions, please open an issue or reach out to the community. 