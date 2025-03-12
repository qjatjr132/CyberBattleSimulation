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
from Onpolicy.ppo.baseline_exp.cyberbattle._env import cyberbattle_env
import numpy as np
from typing import List, NamedTuple, Optional, Tuple, Union
import random

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
import Onpolicy.ppo.baseline_exp.cyberbattle.agents.baseline.agent_wrapper as w
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

    def forward(self, state):
        v = torch.tanh(self.C1(state))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v
    
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


class PPOLearnerPolicy(Learner):

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
                 lambd : float,
                 learning_rate: float
                 ):

        model = CyberBattleStateActionModel(ep)

        linear_input_size = len(model.state_space.dim_sizes)
        self.ppo_t = 0
        self.entropy_coef = 0.01
        self.entropy_coef_decay=0.99
        self.T_horizon = 1024
        self.adv_normalization = True
        self.K_epochs = 5
        self.clip_rate = 0.2
        self.l2_reg=0
        self.state_dim = linear_input_size
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.stateaction_model = CyberBattleStateActionModel(ep)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lambd = lambd
        '''Build Actor and Critic'''
        self.actor = Actor(ep).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic = Critic(ep).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),learning_rate)

        '''Build Trajectory holder'''
        self.s_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.a_hoder = np.zeros((self.T_horizon, 1), dtype=np.int64)
        self.r_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.logprob_a_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.done_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.dw_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)

    def parameters_as_string(self):
        return f'γ={self.gamma}, lr={self.learning_rate}'

    def all_parameters_as_string(self) -> str:
        model = self.stateaction_model
        return f'{self.parameters_as_string()}\n' \
            f'dimension={model.state_space.flat_size()}x{model.action_space.flat_size()}, ' \
            f'Q={[f.name() for f in model.state_space.feature_selection]} ' \
            f"-> 'abstract_action'"

    def optimize_model(self):
        self.entropy_coef *= self.entropy_coef_decay #exploring decay

        '''Prepare PyTorch data from Numpy data'''
        s = torch.from_numpy(self.s_hoder).to(device)
        a = torch.from_numpy(self.a_hoder).to(device)
        r = torch.from_numpy(self.r_hoder).to(device)
        s_next = torch.from_numpy(self.s_next_hoder).to(device)
        old_prob_a = torch.from_numpy(self.logprob_a_hoder).to(device)
        done = torch.from_numpy(self.done_hoder).to(device)
        dw = torch.from_numpy(self.dw_hoder).to(device)

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(device)
            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  #sometimes helps

        """PPO update"""
        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))


        for _ in range(self.K_epochs):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))
                '''actor update'''
                #print('----------')
                prob = self.actor.pi(s[index], softmax_dim=1)
                entropy = Categorical(prob).entropy().sum(0, keepdim=True)
                prob_a = prob.gather(1, a[index])
                #print(f"prob_a : {prob_a}\nold_proba_a = {old_prob_a[index]}")
                ratio = torch.exp(torch.log(prob_a+1e-6) - torch.log(old_prob_a[index]+ 1e-6))  # a/b == exp(log(a)-log(b))
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy
                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                '''critic update'''
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg
                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

    def lookup_ppo(self, states_to_consider: List[ndarray]) -> Tuple[List[np.int32], List[np.int32]]:
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

    def put_data(self, s, a, r, s_next, done,logprob):
        self.s_hoder[self.ppo_t] = s
        self.a_hoder[self.ppo_t] = a
        self.r_hoder[self.ppo_t] = r
        self.s_next_hoder[self.ppo_t] = s_next
        self.done_hoder[self.ppo_t] = done
        self.logprob_a_hoder[self.ppo_t] = logprob

    def update_buffer(self,
                          reward: float,
                          actor_state: ndarray,
                          abstract_action: np.int32,
                          next_actor_state: Optional[ndarray],
                          done: bool,logprob_a
                          ):
        # store the transition in memory

        if next_actor_state is None:
            next_state_tensor = None
            self.put_data(actor_state, np.int_(abstract_action), reward, actor_state, done,logprob_a)
        else:
            next_state_tensor = next_actor_state
            self.put_data(actor_state, np.int_(abstract_action), reward, next_state_tensor, done,logprob_a)
        self.ppo_t+=1
        
        if self.ppo_t != 0 and self.ppo_t % 1024 == 0:
            self.optimize_model()
            self.ppo_t=0

    def on_step(self, wrapped_env: w.AgentWrapper,
                observation, reward: float, done: bool, info, action_metadata,proba):
        agent_state = wrapped_env.state
        if done:
            self.update_buffer(reward,
                                   actor_state=action_metadata.actor_state,
                                   abstract_action=action_metadata.abstract_action,
                                   next_actor_state=None,done=done,logprob_a=proba)
        else:
            next_global_state = self.stateaction_model.global_features.get(agent_state, node=None)
            next_actor_features = self.stateaction_model.node_specific_features.get(
                agent_state, action_metadata.actor_node)
            next_actor_state = self.get_actor_state_vector(next_global_state, next_actor_features)

            self.update_buffer(reward,
                                   actor_state=action_metadata.actor_state,
                                   abstract_action=action_metadata.abstract_action,
                                   next_actor_state=next_actor_state,done=done,logprob_a=proba)
        
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

        remaining_action_lookups, remaining_prob_lookups, m , action_probs, s = self.lookup_ppo(candidate_actor_state_vector)

        remaining_candidate_indices = list(range(len(candidate_actor_state_vector)))
        id = s-1
        while remaining_candidate_indices:
            value, remaining_candidate_index = select_action(remaining_prob_lookups)
  
            actor_index = remaining_candidate_indices[id]
            abstract_action = remaining_action_lookups[remaining_candidate_index]

            actor_features = unique_active_actors_features[actor_index]

            action_style, gym_action, metadata = self.try_exploit_at_candidate_actor_states(
                wrapped_env,
                current_global_state,
                actor_features,
                abstract_action)
            if gym_action:
                return action_style, gym_action, metadata, value
            id -= 1
            remaining_candidate_indices.pop(id)
            remaining_prob_lookups.pop(remaining_candidate_index)
            remaining_action_lookups.pop(remaining_candidate_index)

        return "exploit[undefined]->explore", None, None, 1-value

    def stateaction_as_string(self, action_metadata) -> str:
        return ''