from __future__ import annotations

from math import ceil
from pathlib import Path
from copy import deepcopy
from functools import partial, wraps
from collections import deque, namedtuple
from random import randrange

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, is_tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical
from torch.utils._pytree import tree_map

from torch.nn.utils.rnn import pad_sequence
pad_sequence = partial(pad_sequence, batch_first = True)

import einx
from einops import reduce, repeat, einsum, rearrange, pack, unpack
from einops.layers.torch import Rearrange

from ema_pytorch import EMA

from adam_atan2_pytorch import AdoptAtan2

from hl_gauss_pytorch import HLGaussLoss

from x_transformers import (
    Decoder,
    ContinuousTransformerWrapper
)

from assoc_scan import AssocScan

from accelerate import Accelerator

from x_transformers_rl.evolution import (
    LatentGenePool
)

# memory tuple

Memory = namedtuple('Memory', [
    'state',
    'action',
    'action_log_prob',
    'reward',
    'is_boundary',
    'value',
])

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def frac_gradient(t, frac = 1.):
    assert 0 <= frac <= 1.
    return t.detach() * (1. - frac) + t * frac

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum()

def temp_batch_dim(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        args, kwargs = tree_map(lambda t: rearrange(t, '... -> 1 ...') if is_tensor(t) else t, (args, kwargs))

        out = fn(*args, **kwargs)

        out = tree_map(lambda t: rearrange(t, '1 ... -> ...') if is_tensor(t) else t, out)
        return out

    return inner

def pack_with_inverse(t, pattern):
    packed, shapes = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, shapes, inv_pattern)

    return packed, inverse

def from_numpy(
    t,
    dtype = torch.float32
):
    if is_tensor(t):
        return t

    if isinstance(t, np.float64):
        t = np.array(t)

    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)

    if exists(dtype):
        t = t.type(dtype)

    return t

# world model + actor / critic in one

class WorldModelActorCritic(Module):
    def __init__(
        self,
        transformer: Module,
        num_actions,
        critic_dim_pred,
        critic_min_max_value: tuple[float, float],
        dim_pred_state,
        frac_actor_critic_head_gradient = 0.5,
        entropy_weight = 0.02,
        reward_dropout = 0.5, # dropout the prev reward conditioning half the time, so the world model can still operate without previous rewards
        eps_clip = 0.2,
        value_clip = 0.4
    ):
        super().__init__()
        self.transformer = transformer
        dim = transformer.attn_layers.dim

        self.reward_embed = nn.Parameter(torch.ones(dim) * 1e-2)
        self.action_embeds = nn.Embedding(num_actions, dim)

        self.reward_dropout = nn.Dropout(reward_dropout)

        dim = transformer.attn_layers.dim

        self.to_pred_done = nn.Sequential(
            nn.Linear(dim * 2, 1),
            Rearrange('... 1 -> ...'),
            nn.Sigmoid()            
        )

        self.to_pred = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim_pred_state * 2),
            Rearrange('... (mean_var d) -> mean_var ... d', mean_var = 2)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, critic_dim_pred)
        )

         # https://arxiv.org/abs/2403.03950

        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = critic_min_max_value[0],
            max_value = critic_min_max_value[1],
            num_bins = critic_dim_pred,
            clamp_to_range = True
        )

        self.action_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, num_actions),
            nn.Softmax(dim = -1)
        )

        self.frac_actor_critic_head_gradient = frac_actor_critic_head_gradient

        # ppo loss related

        self.eps_clip = eps_clip
        self.entropy_weight = entropy_weight

        # clipped value loss related

        self.value_clip = value_clip

        self.register_buffer('dummy', tensor(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def compute_autoregressive_loss(
        self,
        pred,
        real
    ):
        pred_mean, pred_var = pred[..., :-1, :] # todo: fix truncation scenario
        return F.gaussian_nll_loss(pred_mean, real[:, 1:], pred_var, reduction = 'none')

    def compute_done_loss(
        self,
        done_pred,
        dones
    ):
        return F.binary_cross_entropy(done_pred, dones.float(), reduction = 'none')

    def compute_actor_loss(
        self,
        action_probs,
        actions,
        old_log_probs,
        returns,
        old_values
    ):
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        scalar_old_values = self.critic_hl_gauss_loss(old_values)

        # calculate clipped surrogate objective, classic PPO loss

        ratios = (action_log_probs - old_log_probs).exp()

        advantages = normalize(returns - scalar_old_values.detach())

        surr1 = ratios * advantages
        surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = - torch.min(surr1, surr2) - self.entropy_weight * entropy
        return actor_loss

    def compute_critic_loss(
        self,
        values,
        returns,
        old_values
    ):
        clip, hl_gauss = self.value_clip, self.critic_hl_gauss_loss

        scalar_old_values = hl_gauss(old_values)
        scalar_values = hl_gauss(values)

        # using the proposal from https://www.authorea.com/users/855021/articles/1240083-on-analysis-of-clipped-critic-loss-in-proximal-policy-gradient

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
        *args,
        actions = None,
        rewards = None,
        next_actions = None,
        **kwargs
    ):
        device = self.device
        sum_embeds = 0.

        if exists(actions):
            has_actions = actions >= 0.
            actions = torch.where(has_actions, actions, 0)
            action_embeds = self.action_embeds(actions)
            action_embeds = einx.where('b n, b n d, ', has_actions, action_embeds, 0.)
            sum_embeds = sum_embeds + action_embeds

        if exists(rewards):
            reward_embeds = einx.multiply('..., d -> ... d', rewards, self.reward_embed)

            maybe_dropout = self.reward_dropout(torch.ones((), device = device)) > 0.

            sum_embeds = sum_embeds + reward_embeds * maybe_dropout.float()

        embed, cache = self.transformer(*args, **kwargs, sum_embeds = sum_embeds, return_embeddings = True, return_intermediates = True)

        # if `next_actions` from agent passed in, use it to predict the next state + truncated / terminated signal

        embed_with_actions = None
        if exists(next_actions):
            next_action_embeds = self.action_embeds(next_actions)
            embed_with_actions = cat((embed, next_action_embeds), dim = -1)

        # predicting state and dones, based on agent's action

        state_pred = None
        dones = None

        if exists(embed_with_actions):
            state_mean, state_log_var = self.to_pred(embed_with_actions)

            state_pred = stack((state_mean, state_log_var.exp()))
            dones = self.to_pred_done(embed_with_actions)

        # actor critic heads living on top of transformer - basically approaching online decision transformer except critic learn discounted returns

        embed = frac_gradient(embed, self.frac_actor_critic_head_gradient) # what fraction of the gradient to pass back to the world model from the actor / critic head

        # actions

        action_probs = self.action_head(embed)

        # values

        values = self.critic_head(embed)

        return action_probs, values, state_pred, dones, cache

# RSM Norm (not to be confused with RMSNorm from transformers)
# this was proposed by SimBa https://arxiv.org/abs/2410.09754
# experiments show this to outperform other types of normalization

class RSMNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        # equation (3) in https://arxiv.org/abs/2410.09754
        super().__init__()
        self.dim = dim
        self.eps = 1e-5

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_variance', torch.ones(dim))

    def forward(
        self,
        x
    ):
        assert x.shape[-1] == self.dim, f'expected feature dimension of {self.dim} but received {x.shape[-1]}'

        time = self.step.item()

        mean = self.running_mean
        variance = self.running_variance

        normed = (x - mean) / variance.sqrt().clamp(min = self.eps)

        if not self.training:
            return normed

        # update running mean and variance

        with torch.no_grad():

            new_obs_mean = reduce(x, '... d -> d', 'mean')
            delta = new_obs_mean - mean

            new_mean = mean + delta / time
            new_variance = (time - 1) / time * (variance + (delta ** 2) / time)

            self.step.add_(1)
            self.running_mean.copy_(new_mean)
            self.running_variance.copy_(new_variance)

        return normed

# GAE

@torch.no_grad()
def calc_gae(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    assert values.shape[-1] == rewards.shape[-1]
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[..., :-1], values[..., 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    returns = gae + values

    return returns

# agent

class Agent(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        reward_range: tuple[float, float],
        epochs,
        max_timesteps,
        batch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        regen_reg_rate,
        cautious_factor,
        eps_clip,
        value_clip,
        ema_decay,
        critic_pred_num_bins = 100,
        hidden_dim = 48,
        world_model: dict = dict(
            attn_dim_head = 16,
            heads = 4,
            depth = 4,
            attn_gate_values = True,
            add_value_residual = True,
            learned_value_residual_mix = True
        ),
        dropout = 0.25,
        max_grad_norm = 0.5,
        frac_actor_critic_head_gradient = 0.5,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1250
        ),
        save_path = './ppo.pt'
    ):
        super().__init__()

        self.model_dim = hidden_dim

        self.model = WorldModelActorCritic(
            num_actions = num_actions,
            critic_dim_pred = critic_pred_num_bins,
            critic_min_max_value = reward_range,
            dim_pred_state = state_dim + 1,
            entropy_weight = beta_s,
            eps_clip = eps_clip,
            value_clip = value_clip,
            transformer = ContinuousTransformerWrapper(
                dim_in = state_dim,
                dim_out = None,
                max_seq_len = max_timesteps,
                probabilistic = True,
                attn_layers = Decoder(
                    dim = hidden_dim,
                    rotary_pos_emb = True,
                    attn_dropout = dropout,
                    ff_dropout = dropout,
                    **world_model
                )
            )
        )

        self.frac_actor_critic_head_gradient = frac_actor_critic_head_gradient

        # state + reward normalization

        self.rsmnorm = RSMNorm(state_dim + 1)

        self.ema_model = EMA(self.model, beta = ema_decay, include_online_model = False, **ema_kwargs)

        self.optimizer = AdoptAtan2(self.model.parameters(), lr = lr, betas = betas, regen_reg_rate = regen_reg_rate, cautious_factor = cautious_factor)

        self.max_grad_norm = max_grad_norm

        self.ema_model.add_to_optimizer_post_step_hook(self.optimizer)

        # learning hparams

        self.batch_size = batch_size

        self.epochs = epochs

        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.save_path = Path(save_path)

        self.register_buffer('dummy', tensor(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def save(self):
        torch.save({
            'model': self.model.state_dict(),
        }, str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path), weights_only = True)

        self.model.load_state_dict(data['model'])

    def learn(self, memories, episode_lens):

        model = self.model
        hl_gauss = self.model.critic_hl_gauss_loss

        # retrieve and prepare data from memory for training - list[list[Memory]]

        def stack_and_to_device(t):
            return stack(t).to(self.device)

        def stack_memories(episode_memories):
            return tuple(map(stack_and_to_device, zip(*episode_memories)))

        memories = map(stack_memories, memories)

        (
            states,
            actions,
            old_log_probs,
            rewards,
            is_boundaries,
            values,
        ) = tuple(map(pad_sequence, zip(*memories)))

        masks = ~is_boundaries

        # calculate generalized advantage estimate

        scalar_values = hl_gauss(values)

        returns = calc_gae(
            rewards = rewards,            
            masks = masks,
            lam = self.lam,
            gamma = self.gamma,
            values = scalar_values,
            use_accelerated = False
        )

        # transformer world model is trained on all states per episode all at once
        # will slowly incorporate other ssl objectives + regularizations from the transformer field

        dataset = TensorDataset(
            states,
            actions,
            rewards,
            old_log_probs,
            returns,
            values,
            is_boundaries,
            episode_lens
        )

        dl = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        model.train()

        rsmnorm_copy = deepcopy(self.rsmnorm) # learn the state normalization alongside in a copy of the state norm module, copy back at the end
        rsmnorm_copy.train()

        for _ in range(self.epochs):
            for (
                states,
                actions,
                rewards,
                old_log_probs,
                returns,
                old_values,
                dones,
                episode_lens
             ) in dl:

                seq = torch.arange(states.shape[1], device = self.device)
                mask = einx.less('n, b -> b n', seq, episode_lens)

                prev_actions = F.pad(actions, (1, -1), value = -1)

                rewards = F.pad(rewards, (1, -1), value = 0.)

                states_with_rewards, inverse_pack = pack_with_inverse((states, rewards), 'b n *')

                with torch.no_grad():
                    self.rsmnorm.eval()
                    states_with_rewards = self.rsmnorm(states_with_rewards)

                states, rewards = inverse_pack(states_with_rewards)

                action_probs, values, states_with_rewards_pred, done_pred, _ = model(
                    states,
                    rewards = rewards,
                    actions = prev_actions,
                    next_actions = actions, # prediction of the next state needs to be conditioned on the agent's chosen action on that state, and will make the world model interactable
                    mask = mask
                )

                # autoregressive loss for transformer world modeling - there's nothing better atm, even if deficient

                world_model_loss = model.compute_autoregressive_loss(
                    states_with_rewards_pred,
                    states_with_rewards
                )

                world_model_loss = world_model_loss[mask[:, :-1]]

                # predicting termination head

                pred_done_loss = model.compute_done_loss(done_pred, dones)
                pred_done_loss = pred_done_loss[mask]

                # update actor and critic

                actor_loss = model.compute_actor_loss(
                    action_probs,
                    actions,
                    old_log_probs,
                    returns,
                    old_values
                )

                critic_loss = model.compute_critic_loss(
                    values,
                    returns,
                    old_values,
                )

                # add world modeling loss + ppo actor / critic loss

                actor_critic_loss = (actor_loss + critic_loss)[mask]

                loss = world_model_loss.mean() + actor_critic_loss.mean() + pred_done_loss.mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

                rsmnorm_copy(states_with_rewards[mask])

        self.rsmnorm.load_state_dict(rsmnorm_copy.state_dict())

    def forward(
        self,
        state,
        reward = None,
        hiddens = None
    ):

        state = from_numpy(state)
        state = state.to(self.device)
        state = rearrange(state, 'd -> 1 1 d')

        has_reward = exists(reward)

        if not has_reward:
            state = F.pad(state, (0, 1), value = 0.)

        with torch.no_grad():
            self.rsmnorm.eval()
            normed_state_with_reward = self.rsmnorm(state)

        normed_state = normed_state_with_reward[..., :-1]

        if has_reward:
            reward = normed_state_with_reward[..., -1:]

        action_probs, *_, next_hiddens = self.model(
            normed_state,
            rewards = reward,
            cache = hiddens,
        )

        action_probs = rearrange(action_probs, '1 1 d -> d')

        return action_probs, next_hiddens

# main

class Learner(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        reward_range,
        world_model: dict,
        max_timesteps = 500,
        batch_size = 8,
        update_episodes = 64,
        lr = 0.0008,
        betas = (0.9, 0.99),
        lam = 0.95,
        gamma = 0.99,
        eps_clip = 0.2,
        value_clip = 0.4,
        beta_s = .01,
        regen_reg_rate = 1e-4,
        cautious_factor = 0.1,
        epochs = 4,
        ema_decay = 0.9,
        save_every = 100,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()

        assert divisible_by(update_episodes, batch_size)

        self.accelerator = Accelerator(**accelerate_kwargs)

        self.agent = Agent(
            state_dim = state_dim,
            num_actions = num_actions,
            reward_range = reward_range,
            world_model = world_model,
            epochs = epochs,
            max_timesteps = max_timesteps,
            batch_size = batch_size,
            lr = lr,
            betas = betas,
            lam = lam,
            gamma = gamma,
            beta_s = beta_s,
            regen_reg_rate = regen_reg_rate,
            cautious_factor = cautious_factor,
            eps_clip = eps_clip,
            value_clip = value_clip,
            ema_decay = ema_decay,
        )

        self.update_episodes = update_episodes

        self.max_timesteps = max_timesteps

        # saving agent

        self.save_every = save_every

        # move to device

        self.to(self.device)

    @property
    def device(self):
        return self.accelerator.device

    def forward(
        self,
        env: object,
        num_episodes: int,
        max_timesteps = None,
        seed = None,
    ):
        max_timesteps = default(max_timesteps, self.max_timesteps)
        num_episodes = ceil(num_episodes / self.update_episodes) * self.update_episodes

        device = self.device

        memories = deque([])
        episode_lens = []

        if exists(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)

        time = 0
        num_policy_updates = 0

        self.agent.eval()
        model = self.agent.ema_model

        for eps in tqdm(range(num_episodes), desc = 'episodes'):

            one_episode_memories = deque([])

            reset_out = env.reset(seed = seed)

            if isinstance(reset_out, tuple):
                state, *_ = reset_out
            else:
                state = reset_out

            state = from_numpy(state).to(device)

            prev_action = tensor(-1).to(device)
            prev_reward = tensor(0.).to(device)

            world_model_cache = None

            @torch.no_grad()
            def state_to_pred_action_and_value(state, prev_action, prev_reward):
                nonlocal world_model_cache

                state_with_reward, inverse_pack = pack_with_inverse((state, prev_reward), '*')

                self.agent.rsmnorm.eval()
                normed_state_reward = self.agent.rsmnorm(state_with_reward)

                normed_state, normed_reward = inverse_pack(normed_state_reward)

                model.eval()

                normed_state = rearrange(normed_state, 'd -> 1 1 d')
                prev_action = rearrange(prev_action, ' -> 1 1')

                action_probs, values, _, _, world_model_cache = model.forward_eval(
                    normed_state,
                    rewards = normed_reward,
                    cache = world_model_cache,
                    actions = prev_action
                )

                action_probs = rearrange(action_probs, '1 1 d -> d')
                values = rearrange(values, '1 1 d -> d')
                return action_probs, values

            for timestep in range(max_timesteps):
                time += 1

                action_probs, value = state_to_pred_action_and_value(state, prev_action, prev_reward)

                dist = Categorical(action_probs)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)

                env_step_out = env.step(action.item())

                if len(env_step_out) >= 4:
                    next_state, reward, terminated, truncated, *_ = env_step_out
                elif len(env_step_out) == 3:
                    next_state, reward, terminated = env_step_out
                    truncated = False
                else:
                    raise RuntimeError('invalid number of returns from environment .step')

                next_state = from_numpy(next_state).to(device)

                reward = float(reward)

                prev_action = action
                prev_reward = tensor(reward).to(device) # from the xval paper, we know pre-norm transformers can handle scaled tokens https://arxiv.org/abs/2310.02989

                memory = Memory(state, action, action_log_prob, tensor(reward), tensor(terminated), value)

                one_episode_memories.append(memory)

                state = next_state

                # determine if truncating or terminated

                done = terminated or truncated

                # take care of truncated by adding a non-learnable memory storing the next value for GAE

                if done and not terminated:
                    _, next_value, *_ = state_to_pred_action_and_value(state, prev_action, prev_reward)

                    bootstrap_value_memory = memory._replace(
                        state = state,
                        is_boundary = tensor(True),
                        value = next_value
                    )

                    memories.append(bootstrap_value_memory)

                # break if done

                if done:
                    break

            episode_lens.append(timestep + 1)

            # add list[Memory] to all episode memories list[list[Memory]]

            memories.append(one_episode_memories)

            # updating of the agent

            if divisible_by(len(memories), self.update_episodes):

                self.agent.learn(memories, tensor(episode_lens, device = device))
                num_policy_updates += 1

                memories.clear()
                episode_lens.clear()

            if divisible_by(eps, self.save_every):
                self.agent.save()


        self.agent.save()

        print(f'training complete')
