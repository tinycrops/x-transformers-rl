import pytest
import numpy as np

@pytest.mark.parametrize('evolutionary', (False, True))
@pytest.mark.parametrize('continuous_actions', (False, True))
def test_e2e(
    evolutionary,
    continuous_actions
):
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
        batch_size = 2,
        update_episodes = 2,
        continuous_actions = continuous_actions,
        evolutionary = evolutionary,
        latent_gene_pool = dict(
            dim = 32,
            num_genes_per_island = 3,
            num_selected = 2,
            tournament_size = 2
        ),
        world_model = dict(
            attn_dim_head = 16,
            heads = 4,
            depth = 1,
        )
    )

    learner(sim, 1)

    # deploying

    agent = learner.agent

    hiddens = None
    actions, hiddens = agent(np.random.randn(5), hiddens = hiddens)
    actions, hiddens = agent(np.random.randn(5), hiddens = hiddens)
