# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#변경 사항
# 1. 경로 자동 생성
# 2. epsilon_greedy_search 함수 내 인자 추가
#   - (start_epi:int (exploration episode 수)
#   - save_count:int (render 저장 빈도)
#   - iteration_info_save = False (iteration information 저장 여부)
#   - trained = False (모델을 불러와서 이어서 학습시킬 시 exploration 제외)
#   - trained_new = True (첫 training 시 exploration 추가))
# 3. success rate 저장을 위한 valid/invalid action의 수 저장
# 4. action에 대한 reward 변경 코드 추가
# 5. 적은 timestep에서 멈췄을 경우 얻는 reward 비율 높이도록 하는 코드 추가 (winning reward 변경)
# 6. 정상적인 학습을 위한 exploration episode 설정 (사용자 지정 가능)
# 7. valid action 상황에서 render 시각화, render:로 수정 시 전체 시각화 가능
# 8. valid action 저장
# 9. 각 episode에 대한 종료 timestep 및 reward 저장

"""Learner helpers and epsilon greedy search"""
import math
import sys
import torch
import os
from .plotting import PlotTraining, plot_averaged_cummulative_rewards
from .agent_wrapper import AgentWrapper, EnvironmentBounds, Verbosity, ActionTrackingStateAugmentation
import logging
import numpy as np
from cyberbattle._env import cyberbattle_env
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, TypedDict, List
import progressbar
import abc
import pandas as pd
from PIL import Image
import torch
import joblib
import time


class Learner(abc.ABC):
    """Interface to be implemented by an epsilon-greedy learner"""

    def new_episode(self) -> None:
        return None

    def end_of_episode(self, i_episode, t) -> None:
        return None

    def end_of_iteration(self, t, done) -> None:
        return None

    @abc.abstractmethod
    def explore(self, wrapped_env: AgentWrapper) -> Tuple[str, cyberbattle_env.Action, object]:
        """Exploration function.
        Returns (action_type, gym_action, action_metadata) where
        action_metadata is a custom object that gets passed to the on_step callback function"""
        raise NotImplementedError

    @abc.abstractmethod
    def exploit(self, wrapped_env: AgentWrapper, observation) -> Tuple[str, Optional[cyberbattle_env.Action], object]:
        """Exploit function.
        Returns (action_type, gym_action, action_metadata) where
        action_metadata is a custom object that gets passed to the on_step callback function"""
        raise NotImplementedError

    @abc.abstractmethod
    def on_step(self, wrapped_env: AgentWrapper, observation, reward, done, info, action_metadata) -> None:
        raise NotImplementedError

    def parameters_as_string(self) -> str:
        return ''

    def all_parameters_as_string(self) -> str:
        return ''

    def loss_as_string(self) -> str:
        return ''

    def stateaction_as_string(self, action_metadata) -> str:
        return ''


class RandomPolicy(Learner):
    """A policy that does not learn and only explore"""

    def explore(self, wrapped_env: AgentWrapper) -> Tuple[str, cyberbattle_env.Action, object]:
        gym_action = wrapped_env.env.sample_valid_action()
        return "explore", gym_action, None

    def exploit(self, wrapped_env: AgentWrapper, observation) -> Tuple[str, Optional[cyberbattle_env.Action], object]:
        raise NotImplementedError

    def on_step(self, wrapped_env: AgentWrapper, observation, reward, done, info, action_metadata):
        return None


Breakdown = TypedDict('Breakdown', {
    'local': int,
    'remote': int,
    'connect': int
})

Outcomes = TypedDict('Outcomes', {
    'reward': Breakdown,
    'noreward': Breakdown
})

Stats = TypedDict('Stats', {
    'exploit': Outcomes,
    'explore': Outcomes,
    'exploit_deflected_to_explore': int
})

TrainedLearner = TypedDict('TrainedLearner', {
    'all_episodes_rewards': List[List[float]],
    'all_episodes_availability': List[List[float]],
    'learner': Learner,
    'trained_on': str,
    'title': str
})

def print_stats(stats):
    """Print learning statistics"""
    def print_breakdown(stats, actiontype: str):
        def ratio(kind: str) -> str:
            x, y = stats[actiontype]['reward'][kind], stats[actiontype]['noreward'][kind]
            sum = x + y
            if sum == 0:
                return 'NaN'
            else:
                return f"{(x / sum):.2f}"

        def print_kind(kind: str):
            print(
                f"    {actiontype}-{kind}: {stats[actiontype]['reward'][kind]}/{stats[actiontype]['noreward'][kind]} "
                f"({ratio(kind)})")
        print_kind('local')
        print_kind('remote')
        print_kind('connect')

    print("  Breakdown [Reward/NoReward (Success rate)]")
    print_breakdown(stats, 'explore')
    print_breakdown(stats, 'exploit')
    print(f"  exploit deflected to exploration: {stats['exploit_deflected_to_explore']}")

#epsilon_greedy_search 함수 내 인자 추가 (start_epi:int, save_count:int, iteration_info_save = False, trained = False, trained_new = True)
def epsilon_greedy_search(
    cyberbattle_gym_env: cyberbattle_env.CyberBattleEnv,
    environment_properties: EnvironmentBounds,
    learner: Learner,
    title: str,
    env_str: str,
    episodes: int,
    step: int,
    iteration_count: int,
    save_count:int,
    epsilon: float,
    epsilon_minimum=0.0,
    epsilon_multdecay: Optional[float] = None,
    epsilon_exponential_decay: Optional[int] = None,
    render=False,
    render_last_episode_rewards_to: Optional[str] = None,
    verbosity: Verbosity = Verbosity.Normal,
    plot_episodes_length=False,

) -> TrainedLearner:
    """Epsilon greedy search for CyberBattle gym environments

    Parameters
    ==========

    - cyberbattle_gym_env -- the CyberBattle environment to train on

    - learner --- the policy learner/exploiter

    - episode_count -- Number of training episodes

    - iteration_count -- Maximum number of iterations in each episode

    - epsilon -- explore vs exploit
        - 0.0 to exploit the learnt policy only without exploration
        - 1.0 to explore purely randomly

    - epsilon_minimum -- epsilon decay clipped at this value.
    Setting this value too close to 0 may leed the search to get stuck.

    - epsilon_decay -- epsilon gets multiplied by this value after each episode

    - epsilon_exponential_decay - if set use exponential decay. The bigger the value
    is, the slower it takes to get from the initial `epsilon` to `epsilon_minimum`.

    - verbosity -- verbosity of the `print` logging

    - render -- render the environment interactively after each episode

    - render_last_episode_rewards_to -- render the environment to the specified file path
    with an index appended to it each time there is a positive reward
    for the last episode only

    - plot_episodes_length -- Plot the graph showing total number of steps by episode
    at th end of the search.

    Note on convergence
    ===================

    Setting 'minimum_espilon' to 0 with an exponential decay <1
    makes the learning converge quickly (loss function getting to 0),
    but that's just a forced convergence, however, since when
    epsilon approaches 0, only the q-values that were explored so
    far get updated and so only that subset of cells from
    the Q-matrix converges.
    """
    dir = os.listdir(f'./result/timestep/')
    os.rmdir(f'./result/timestep/{int(dir[0])}')
    os.makedirs('./result/timestep/0')


    # add- model path
    log_dir = f"./tensorboard/log_dir_{title}_{env_str}"
    writer = SummaryWriter(log_dir)
    j=0
    while True:

        #경로 자동 생성
        if os.path.exists(f'./result/{title}/{env_str}/v{j}/'):
            j += 1
        else:
            saverender_path = f'./result/{title}/{env_str}/v{j}/'
            os.makedirs(saverender_path)
            os.makedirs(saverender_path+'model')
            os.makedirs(saverender_path+'action')
            break

    savemodel_path = f'./result/{title}/{env_str}/v{j}/model/{title}.pkl'

    print(f"###### {title}\n"
          f"Learning with: iteration_count={iteration_count},"
          f"ϵ={epsilon},"
          f'ϵ_min={epsilon_minimum}, '
          + (f"ϵ_multdecay={epsilon_multdecay}," if epsilon_multdecay else '')
          + (f"ϵ_expdecay={epsilon_exponential_decay}," if epsilon_exponential_decay else '') +
          f"{learner.parameters_as_string()}")

    initial_epsilon = epsilon
    # add-draw reward-episode graph
    reward_per_episodes = []

    # add- save success rate dataset
    action_list = ['explore', 'exploit']
    kind = ['local', 'remote', 'connect']
    reward_dict = {'explore_local': [], 'explore_remote': [], 'explore_connect': [], 'exploit_local': [],
    'exploit_remote': [], 'exploit_connect': []}
    noreward_dict = {'explore_local': [], 'explore_remote': [], 'explore_connect': [], 'exploit_local': [],
    'exploit_remote': [], 'exploit_connect': []}

    all_episodes_rewards = []
    all_episodes_availability = []
    episode_per_reward = []
    episode_per_ended_t = []
    success_rate = []
    success_rate_exploit = []
    ended_step = []
    wrapped_env = AgentWrapper(cyberbattle_gym_env,
                               ActionTrackingStateAugmentation(environment_properties, cyberbattle_gym_env.reset()))

    plot_title = f"{title} (iteration={iteration_count}, ϵ={initial_epsilon}, ϵ_min={epsilon_minimum}," \
        + (f"ϵ_multdecay={epsilon_multdecay}," if epsilon_multdecay else '') \
        + (f"ϵ_expdecay={epsilon_exponential_decay}," if epsilon_exponential_decay else '') \
        + learner.parameters_as_string()
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    bar = progressbar.ProgressBar(
        widgets=[
            'Total timestep ',
            f'{iteration_count}',
            '|Iteration ',
            progressbar.Counter(),
            '|',
            progressbar.Variable(name='reward', width=6, precision=10),
            '|',
            progressbar.Variable(name='last_reward_at', width=4),
            '|',
            progressbar.Timer(),
            progressbar.Bar()
        ],
        redirect_stdout=False)
        
    episode = 0
    steps_done = 0    
    ppo_t = 0
    
    while episode < episodes+1:

        episode += 1
        ended_t=0
        steps = 0
        print(f"  ## Episode: {episode} '{title}' "
              f"ϵ={epsilon:.4f}, "
              f"{learner.parameters_as_string()},"
              f"current ϵ={epsilon}")
        if episode % save_count == 0:
            if not os.path.exists(saverender_path + f'episode {episode}/'):
                os.makedirs(saverender_path+f'episode {episode}/render')

        observation = wrapped_env.reset()
        total_reward = 0.0
        all_rewards = []
        all_availability = []

        learner.new_episode()

        stats = Stats(exploit=Outcomes(reward=Breakdown(local=0, remote=0, connect=0),
                                       noreward=Breakdown(local=0, remote=0, connect=0)),
                      explore=Outcomes(reward=Breakdown(local=0, remote=0, connect=0),
                                       noreward=Breakdown(local=0, remote=0, connect=0)),
                      exploit_deflected_to_explore=0
                      )

        episode_ended_at = None
        sys.stdout.flush()

        path_action = saverender_path+f'action/{title}_e{episode}_action.txt'
        actor_reward = 0
        #정상적인 학습을 위한 exploration episode 설정 (사용자 지정 가능)
        while True:
         
            steps += 1
            steps_done += 1
	    # exploration episode 설정

            action_style, gym_action, action_metadata= learner.exploit(wrapped_env, observation)
            if not gym_action:
                stats['exploit_deflected_to_explore'] += 1
                _, gym_action, action_metadata = learner.explore(wrapped_env)
            
            logging.debug(f"timestep {steps} : gym_action={gym_action}, action_metadata={action_metadata}\n")
            
            observation, reward, done, info = wrapped_env.step(gym_action)

            action_type = 'exploit' if action_style == 'exploit' else 'explore'
            outcome = 'reward' if reward > 0 else 'noreward'
            if 'local_vulnerability' in gym_action:
                stats[action_type][outcome]['local'] += 1
            elif 'remote_vulnerability' in gym_action:
                stats[action_type][outcome]['remote'] += 1
            else:
                stats[action_type][outcome]['connect'] += 1
            actor_reward += reward
            learner.on_step(wrapped_env, observation, reward, done, info, action_metadata)
            assert np.shape(reward) == ()

            all_rewards.append(reward)
            all_availability.append(info['network_availability'])
            total_reward += reward
            bar.update(steps_done, reward=total_reward)

            if reward > 0 :
                bar.update(steps_done, last_reward_at=steps)

            if verbosity == Verbosity.Verbose or (verbosity == Verbosity.Normal and reward>0):
                sign = ['-', '+'][reward > 0]

                print(f"    {sign} t={steps_done} {action_style} r={reward} cum_reward:{total_reward} "
                      f"a={action_metadata}-{gym_action} "
                      f"creds={len(observation['credential_cache_matrix'])} "
                      f" {learner.stateaction_as_string(action_metadata)}")

            learner.end_of_iteration(steps, done)

            if render and (episode % save_count ==0):
                fig = cyberbattle_gym_env.render_as_fig()
                fig.write_image(saverender_path + f'episode {episode}/render'+f'/{str(steps).zfill(4)}.png')

            if steps == step:
                done = True

            if done:
                episode_ended_at = ended_t
                bar.finish(dirty=True)
                break

        episodes_success_rate = (stats['exploit']['reward']['local'] + 
                                 stats['exploit']['reward']['remote'] + 
                                 stats['exploit']['reward']['connect'] + 
                                 stats['explore']['reward']['local'] + 
                                 stats['explore']['reward']['remote'] + 
                                 stats['explore']['reward']['connect'])/steps

        if (stats['exploit']['reward']['local'] + stats['exploit']['reward']['remote'] + stats['exploit']['reward']['connect'] + stats['exploit']['noreward']['local'] + stats['exploit']['noreward']['remote'] + stats['exploit']['noreward']['connect']) == 0:
            episodes_success_rate_exploit = 0

        else:
            episodes_success_rate_exploit = (stats['exploit']['reward']['local'] + 
                                 stats['exploit']['reward']['remote'] + 
                                 stats['exploit']['reward']['connect'])/ (stats['exploit']['reward']['local'] + 
                                 stats['exploit']['reward']['remote'] + 
                                 stats['exploit']['reward']['connect'] +
                                 stats['exploit']['noreward']['local'] + 
                                 stats['exploit']['noreward']['remote'] + 
                                 stats['exploit']['noreward']['connect'])

                                 
        writer.add_scalar("episode", episode, steps_done)
        writer.add_scalar("epsilon", epsilon, steps_done)
        writer.add_scalar("episode_return", sum(all_rewards), steps_done)
        writer.add_scalar("episode_steps", steps, steps_done)

        sys.stdout.flush()
        episode_per_reward.append(sum(all_rewards))
        episode_per_ended_t.append(steps)
        ended_step.append(steps)
        success_rate.append(episodes_success_rate)
        success_rate_exploit.append(episodes_success_rate_exploit)
        loss_string = learner.loss_as_string()
        if loss_string:
            loss_string = "loss={loss_string}"

        if episode_ended_at:
            print(f"  Episode {episode} ended at t={steps} {loss_string}")
        else:
            print(f"  Episode {episode} stopped at t={steps} {loss_string}")

        print_stats(stats)
        reward_per_episodes.append(total_reward)
        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)

        writer.add_scalar('rewards', sum(all_rewards), episode)
        writer.add_scalar('ended timestep', steps if episode_ended_at else iteration_count, episode)

        length = episode_ended_at if episode_ended_at else iteration_count

        if plot_episodes_length:
            plottraining.episode_done(length)

        #if render:
        #    wrapped_env.render()

        if epsilon_multdecay:
            epsilon = max(epsilon_minimum, epsilon * epsilon_multdecay)

        # add - make dataset dictionary for success rate
        '''
        # success rate 저장을 위한 valid/invalid action의 수 저장
        for at in action_list:
            for k in kind:
                reward_dict[f'{at}_{k}'].append(stats[at]['reward'][k])
                noreward_dict[f'{at}_{k}'].append(stats[at]['noreward'][k])
        '''
    #wrapped_env.close()
    
    print("simulation ended")
    if plot_episodes_length:
             plottraining.plot_end()
    
    np.save(f"./{title}_success_rate_{time.strftime('%y%m%d - %X')}.npy", success_rate)
    np.save(f"./{title}_episode_per_reward_{time.strftime('%y%m%d - %X')}.npy", episode_per_reward)
    np.save(f"./{title}_episode_per_step_{time.strftime('%y%m%d - %X')}.npy", ended_step)
    np.save(f"./{title}_success_rate_exploit_{time.strftime('%y%m%d - %X')}.npy", success_rate_exploit)
    
    with open(f'./{title}_cummulative_reward_list.txt', 'w') as f:
        for row in all_episodes_rewards:
            f.write(' '.join(map(str, row)) + '\n')

    # add return reward per epi, change title name
    return TrainedLearner(
        reward_per_episodes=reward_per_episodes,
        all_episodes_rewards=all_episodes_rewards,
        all_episodes_availability=all_episodes_availability,
        learner=learner,
        trained_on=cyberbattle_gym_env.name,
        title=title,
        model_path = savemodel_path
    )

def transfer_learning_evaluation(
    environment_properties: EnvironmentBounds,
    trained_learner: TrainedLearner,
    eval_env: cyberbattle_env.CyberBattleEnv,
    eval_epsilon: float,
    eval_episode_count: int,
    iteration_count: int,
    benchmark_policy: Learner = RandomPolicy(),
    benchmark_training_args=dict(title="Benchmark", epsilon=1.0)
):
    """Evaluated a trained agent on another environment of different size"""

    eval_oneshot_all = epsilon_greedy_search(
        eval_env,
        environment_properties,
        learner=trained_learner['learner'],
        episode_count=eval_episode_count,  # one shot from learnt Q matric
        iteration_count=iteration_count,
        epsilon=eval_epsilon,
        render=False,
        verbosity=Verbosity.Quiet,
        title=f"One shot on {eval_env.name} - Trained on {trained_learner['trained_on']}",
    )

    eval_random = epsilon_greedy_search(
        eval_env,
        environment_properties,
        learner=benchmark_policy,
        episode_count=eval_episode_count,
        iteration_count=iteration_count,
        verbosity=Verbosity.Quiet,
        **benchmark_training_args
    )

    plot_averaged_cummulative_rewards(
        all_runs=[eval_oneshot_all, eval_random],
        title=f"Transfer learning {trained_learner['trained_on']}->{eval_env.name} "
        f'-- max_nodes={environment_properties.maximum_node_count}, '
        f'episodes={eval_episode_count},\n'
        f"{trained_learner['learner'].all_parameters_as_string()}")
