## x-transformers-rl (wip)

Implementation of a transformer for reinforcement learning using `x-transformers`

## Install

```bash
$ pip install x-transformers-rl
```

## Usage

```python
import numpy as np

class Sim:
    def reset(self, seed = None):
        return np.random.randn(5) # state

    def step(self, actions):
        return np.random.randn(5), np.random.randn(1), False # state, reward, done

sim = Sim()

# learning

from x_transformers_rl import Learner

learner = Learner(
    state_dim = 5,
    num_actions = 2,
    reward_range = (-1., 1.),
    max_timesteps = 10,
    world_model = dict(
        attn_dim_head = 16,
        heads = 4,
        depth = 1,
    )
)

learner(sim, 100)
```

## Example

### Lunar Lander

```bash
$ pip install -r requirements.txt
```

Then

```python
$ python train_lander.py
```

## Citation

```bibtex
@inproceedings{Wang2025EvolutionaryPO,
    title = {Evolutionary Policy Optimization},
    author = {Jianren Wang and Yifan Su and Abhinav Gupta and Deepak Pathak},
    year  = {2025},
    url   = {https://api.semanticscholar.org/CorpusID:277313729}
}
```
