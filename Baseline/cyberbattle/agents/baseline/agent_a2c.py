# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Function DeepQLearnerPolicy.optimize_model:
#   Copyright (c) 2017, Pytorch contributors
#   All rights reserved.
#   https://github.com/pytorch/tutorials/blob/master/LICENSE

"""Deep Q-learning agent applied to chain network (notebook)
This notebooks can be run directly from VSCode, to generate a
traditional Jupyter Notebook to open in your browser
 you can run the VSCode command `Export Currenty Python File As Jupyter Notebook`.

Requirements:
    Nvidia CUDA drivers for WSL2: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
    PyTorch
"""

# pylint: disable=invalid-name

# %% [markdown]
# # Chain network CyberBattle Gym played by a Deeo Q-learning agent

# %%
from numpy import ndarray
from cyberbattle._env import cyberbattle_env
import numpy as np
from typing import List, NamedTuple, Optional, Tuple, Union
import random
from torch.autograd import Variable
# deep learning packages
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torch.cuda
from torch.nn.utils.clip_grad import clip_grad_norm_
import os
from .learner import Learner
from .agent_wrapper import EnvironmentBounds
import cyberbattle.agents.baseline.agent_wrapper as w
from .agent_randomcredlookup import CredentialCacheExploiter
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import numpy as np
import copy
import math
from gym import spaces
from gym.spaces.space import Space



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_dir = "./tensorboard/DQNLoss"
writer = SummaryWriter(log_dir)

class CyberBattleStateActionModel:
    """ Define an abstraction of the state and action space
        for a CyberBattle environment, to be used to train a Q-function.
    """

    def __init__(self, ep: EnvironmentBounds):
        self.ep = ep

        self.global_features = w.ConcatFeatures(ep, [
            w.Feature_discovered_node_count(ep),
            w.Feature_owned_node_count(ep),
            w.Feature_discovered_notowned_node_count(ep, None),

            # w.Feature_discovered_ports(ep),
            # w.Feature_discovered_ports_counts(ep),
            w.Feature_discovered_ports_sliding(ep),
            # w.Feature_discovered_credential_count(ep),
            w.Feature_discovered_nodeproperties_sliding(ep),
        ])

        self.node_specific_features = w.ConcatFeatures(ep, [
            w.Feature_actions_tried_at_node(ep),
            #w.Feature_success_actions_at_node(ep),
            #w.Feature_failed_actions_at_node(ep),
            w.Feature_active_node_properties(ep),
            w.Feature_active_node_age(ep)
            # w.Feature_active_node_id(ep)
        ])

        self.state_space = w.ConcatFeatures(ep, self.global_features.feature_selection +
                                            self.node_specific_features.feature_selection)

        self.action_space = w.AbstractAction(ep)

    def get_state_astensor(self, state: w.StateAugmentation):
        state_vector = self.state_space.get(state, node=None)
        state_vector_float = np.array(state_vector, dtype=np.float32)
        state_tensor = torch.from_numpy(state_vector_float).unsqueeze(0)
        return state_tensor

    def implement_action(
            self,
            wrapped_env: w.AgentWrapper,
            actor_features: ndarray,
            abstract_action: np.int32) -> Tuple[str, Optional[cyberbattle_env.Action], Optional[int]]:
        """Specialize an abstract model action into a CyberBattle gym action.

            actor_features -- the desired features of the actor to use (source CyberBattle node)
            abstract_action -- the desired type of attack (connect, local, remote).

            Returns a gym environment implementing the desired attack at a node with the desired embedding.
        """

        observation = wrapped_env.state.observation

        # Pick source node at random (owned and with the desired feature encoding)
        potential_source_nodes = [
            from_node
            for from_node in w.owned_nodes(observation)
            if np.all(actor_features == self.node_specific_features.get(wrapped_env.state, from_node))
        ]

        if len(potential_source_nodes) > 0:
            source_node = np.random.choice(potential_source_nodes)

            gym_action = self.action_space.specialize_to_gymaction(
                source_node, observation, np.int32(abstract_action))

            if not gym_action:
                return "exploit[undefined]->explore", None, None

            elif wrapped_env.env.is_action_valid(gym_action, observation['action_mask']):
                return "exploit", gym_action, source_node
            else:
                return "exploit[invalid]->explore", None, None
        else:
            return "exploit[no_actor]->explore", None, None

# %%

# PPO

class Actor(nn.Module):
    def __init__(self, ep: EnvironmentBounds):
        super(Actor, self).__init__()

        model = CyberBattleStateActionModel(ep)

        linear_input_size = len(model.state_space.dim_sizes)
        output_size = model.action_space.flat_size()
        self.l1 = nn.Linear(linear_input_size, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, output_size)

        self.policy_history = Variable(torch.Tensor()).to(device)
        self.reward_episode = []

        self.reward_history = []
        self.loss_history = []


    def forward(self, state):

        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob

class Critic(nn.Module):
    def __init__(self, ep: EnvironmentBounds):
        super(Critic, self).__init__()

        model = CyberBattleStateActionModel(ep)

        linear_input_size = len(model.state_space.dim_sizes)

        self.C1 = nn.Linear(linear_input_size, 256)
        self.C2 = nn.Linear(256, 128)
        self.C3 = nn.Linear(128, 1)

        self.value_episode = []
        self.value_history = Variable(torch.Tensor()).to(device)

    def forward(self, state):
        v = torch.tanh(self.C1(state))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v
    
    # combined module (mostly for loading / storing)
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super(ActorCritic, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x):
        value = self.critic(x)
        policy = self.actor(x)

        return value, policy
    
def select_action(array):
    """Just like `argmax` but if there are multiple elements with the max
    return a random index to break ties instead of returning the first one."""
    index = random.choices(range(len(array)), weights = array)
    value = array[index[0]]
    #max_index = np.where(array == value)[0]

    return value, index[0]

class ChosenActionMetadata(NamedTuple):
    """Additonal info about the action chosen by the DQN-induced policy"""
    abstract_action: np.int32
    actor_node: int
    actor_features: ndarray
    actor_state: ndarray

    def __repr__(self) -> str:
        return f"[abstract_action={self.abstract_action}, actor={self.actor_node}, state={self.actor_state}]"


class A2CLearnerPolicy(Learner):

    """Deep Q-Learning on CyberBattle environments

    Parameters
    ==========
    ep -- global parameters of the environment
    model -- define a state and action abstraction for the gym environment
    gamma -- Q discount factor
    replay_memory_size -- size of the replay memory
    batch_size    -- Deep Q-learning batch
    target_update -- Deep Q-learning replay frequency (in number of episodes)
    learning_rate -- the learning rate

    Parameters from DeepDoubleQ paper
        - learning_rate = 0.00025
        - linear epsilon decay
        - gamma = 0.99

    Pytorch code from tutorial at
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self,
                 ep: EnvironmentBounds,
                 gamma: float,
                 batch_size: int,
                 learning_rate: float
                 ):

        model = CyberBattleStateActionModel(ep)

        linear_input_size = len(model.state_space.dim_sizes)
        self.ppo_t = 0
        self.entropy = 0
        self.state_dim = linear_input_size
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.stateaction_model = CyberBattleStateActionModel(ep)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate

        '''Build Actor and Critic'''
        self.actor = Actor(ep).to(device)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic = Critic(ep).to(device)
        self.c_opt = torch.optim.Adam(self.critic.parameters(),learning_rate)

    def estimate_value(self, state):
        pred = self.critic(state).squeeze(0)
        if self.critic.value_history.dim() != 0:
            self.critic.value_history = torch.cat([self.critic.value_history, pred])
        else:
            self.critic.policy_history = (pred)

    def parameters_as_string(self):
        return f'γ={self.gamma}, lr={self.learning_rate}'

    def all_parameters_as_string(self) -> str:
        model = self.stateaction_model
        return f'{self.parameters_as_string()}\n' \
            f'dimension={model.state_space.flat_size()}x{model.action_space.flat_size()}, ' \
            f'Q={[f.name() for f in model.state_space.feature_selection]} ' \
            f"-> 'abstract_action'"

    def optimize_model(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        q_vals = []

        # "unroll" the rewards, apply gamma
        for r in self.actor.reward_episode[::-1]:
            R = r + self.gamma * R
            q_vals.insert(0, R)

        q_vals = torch.FloatTensor(q_vals).to(device)
        values = self.critic.value_history
        log_probs = self.actor.policy_history

        # print(values)
        # print(log_probs)
        advantage = q_vals - values

        self.c_opt.zero_grad()
        critic_loss = 0.0005 * advantage.pow(2).mean()
        critic_loss.requires_grad_(True)
        critic_loss.backward()
        self.c_opt.step()

        self.a_opt.zero_grad()
        actor_loss = (-log_probs * advantage.detach()).mean() + 0.001 * self.entropy

        actor_loss.requires_grad_(True)
        actor_loss.backward()
        self.a_opt.step()

        self.actor.reward_episode = []
        self.actor.policy_history = Variable(torch.Tensor()).to(device)
        self.critic.value_history = Variable(torch.Tensor()).to(device)

    def lookup_a2c(self, states_to_consider: List[ndarray]) -> Tuple[List[np.int32], List[np.int32]]:
        """ Given a set of possible current states return:
            - index, in the provided list, of the state that would yield the best possible outcome
            - the best action to take in such a state"""
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # action: np.int32 = self.policy_net(states_to_consider).max(1)[1].view(1, 1).item()

            state_batch = torch.tensor(states_to_consider).to(device)
            s = state_batch.shape[0]
            '''
            dnn_output = self.actor(state_batch).max(1)
            action_lookups = dnn_output[1].tolist()
            expectedq_lookups = dnn_output[0].tolist()
            '''
            state = state_batch[state_batch.shape[0]-1]
            action_probs = self.actor.pi(state)

            m = Categorical(action_probs)
            
            #max_action_probs, action_indices = action_probs
            
            
            action_probs_a = action_probs.tolist()
            action_index = list(range(len(action_probs_a)))
        
        return action_index, action_probs_a, m , action_probs, s
        '''
    def select_action(self, state):
        probs = self.actor.pi(state)
        action_probs = self.actor.pi(state)
        max_action_probs, action_indices = action_probs.max(dim=1)
        
        max_action_probs = max_action_probs.tolist()
        action_indices = action_indices.tolist()
        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)
        # and sample an action using the distribution
        action = m.sample()
        # place log probabilities into the policy history log\pi(a | s)
        if self.actor.policy_history.dim() != 0:
            self.actor.policy_history = torch.cat([self.actor.policy_history, m.log_prob(action)])
        else:
            self.actor.policy_history = m.log_prob(action)
        return action.item()
        '''
    def on_step(self, wrapped_env: w.AgentWrapper,
                observation, reward: float, done: bool, info, action_metadata):
        self.actor.reward_episode.append(reward)
        self.optimize_model()

    def get_actor_state_vector(self, global_state: ndarray, actor_features: ndarray) -> ndarray:
        return np.concatenate((np.array(global_state, dtype=np.float32),
                               np.array(actor_features, dtype=np.float32)))

    def metadata_from_gymaction(self, wrapped_env, gym_action):
        current_global_state = self.stateaction_model.global_features.get(wrapped_env.state, node=None)
        actor_node = cyberbattle_env.sourcenode_of_action(gym_action)
        actor_features = self.stateaction_model.node_specific_features.get(wrapped_env.state, actor_node)
        abstract_action = self.stateaction_model.action_space.abstract_from_gymaction(gym_action)
        return ChosenActionMetadata(
            abstract_action=abstract_action,
            actor_node=actor_node,
            actor_features=actor_features,
            actor_state=self.get_actor_state_vector(current_global_state, actor_features))

    def explore(self, wrapped_env: w.AgentWrapper
                ) -> Tuple[str, cyberbattle_env.Action, object]:
        """Random exploration that avoids repeating actions previously taken in the same state"""
        # sample local and remote actions only (excludes connect action)
        gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])
        metadata = self.metadata_from_gymaction(wrapped_env, gym_action)
        return "explore", gym_action, metadata

    def try_exploit_at_candidate_actor_states(
            self,
            wrapped_env,
            current_global_state,
            actor_features,
            abstract_action):

        actor_state = self.get_actor_state_vector(current_global_state, actor_features)

        action_style, gym_action, actor_node = self.stateaction_model.implement_action(
            wrapped_env, actor_features, abstract_action)

        if gym_action:
            assert actor_node is not None, 'actor_node should be set together with gym_action'

            return action_style, gym_action, ChosenActionMetadata(
                abstract_action=abstract_action,
                actor_node=actor_node,
                actor_features=actor_features,
                actor_state=actor_state)
        else:
            '''
            # learn the failed exploit attempt in the current state
            self.update_q_function(reward=0.0,
                                   actor_state=actor_state,
                                   next_actor_state=actor_state,
                                   abstract_action=abstract_action)
            '''
            return "exploit[undefined]->explore", None, None

    def exploit(self,
                wrapped_env,
                observation
                ) -> Tuple[str, Optional[cyberbattle_env.Action], object]:

        # first, attempt to exploit the credential cache
        # using the crecache_policy
        # action_style, gym_action, _ = self.credcache_policy.exploit(wrapped_env, observation)
        # if gym_action:
        #     return action_style, gym_action, self.metadata_from_gymaction(wrapped_env, gym_action)

        # Otherwise on exploit learnt Q-function

        current_global_state = self.stateaction_model.global_features.get(wrapped_env.state, node=None)

        # Gather the features of all the current active actors (i.e. owned nodes)
        active_actors_features: List[ndarray] = [
            self.stateaction_model.node_specific_features.get(wrapped_env.state, from_node)
            for from_node in w.owned_nodes(observation)
        ]
        unique_active_actors_features: List[ndarray] = list(np.unique(active_actors_features, axis=0))

        # array of actor state vector for every possible set of node features
        #action, prob = self.select_action(candidate_actor_state_vector)

        candidate_actor_state_vector: List[ndarray] = [
            self.get_actor_state_vector(current_global_state, node_features)
            for node_features in unique_active_actors_features]
        
        state_batch = torch.tensor(candidate_actor_state_vector).to(device)
        
        self.estimate_value(state_batch)
        
        remaining_action_lookups, remaining_prob_lookups, m , action_probs, s= self.lookup_a2c(candidate_actor_state_vector)

        remaining_candidate_indices = list(range(len(candidate_actor_state_vector)))
        id = s-1
        while remaining_candidate_indices:
            _, remaining_candidate_index = select_action(remaining_prob_lookups)

            actor_index = remaining_candidate_indices[id]
            abstract_action = remaining_action_lookups[remaining_candidate_index]

            actor_features = unique_active_actors_features[actor_index]

            action_style, gym_action, metadata = self.try_exploit_at_candidate_actor_states(
                wrapped_env,
                current_global_state,
                actor_features,
                abstract_action)
            
            if gym_action:
                if self.actor.policy_history.numel() != 0:
                    self.actor.policy_history = torch.cat([self.actor.policy_history, m.log_prob(torch.tensor(abstract_action, dtype=torch.int32).cuda())])
                else:
                    self.actor.policy_history = m.log_prob(torch.tensor(abstract_action, dtype=torch.int32).cuda())
                
                p = action_probs.cpu().detach().numpy()
                e = -np.sum(np.mean(p, dtype=np.float64) * np.log1p(p))
                self.entropy += e
                return action_style, gym_action, metadata

            remaining_candidate_indices.pop(id)
            remaining_prob_lookups.pop(remaining_candidate_index)
            remaining_action_lookups.pop(remaining_candidate_index)
            id -= 1
        if self.actor.policy_history.numel() != 0:
            self.actor.policy_history = torch.cat([self.actor.policy_history,1- m.log_prob(torch.tensor(abstract_action, dtype=torch.int32).cuda())])
        else:
            self.actor.policy_history = 1 - m.log_prob(torch.tensor(abstract_action, dtype=torch.int32).cuda())


        return "exploit[undefined]->explore", None, None

    def stateaction_as_string(self, action_metadata) -> str:
        return ''
    def stateaction_as_string(self, action_metadata) -> str:
        return ''