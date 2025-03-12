import sys
import logging
import gym
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_tabularqlearning as tqa
import cyberbattle.agents.baseline.agent_DuelingDQN as ddqla
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_drqn2 as drqla2
import cyberbattle.agents.baseline.agent_tabularqlearning as tqa
from cyberbattle.agents.baseline.agent_wrapper import Verbosity

from cyberbattle.agents.baseline.agent_wrapper import Verbosity
from cyberbattle._env.defender import ScanAndReimageCompromisedMachines
from cyberbattle._env.cyberbattle_env import AttackerGoal, CyberBattleEnv, DefenderConstraint
from typing import cast
import joblib
import os
logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

gymid = "CyberBattleToyCtf-v0"
#env_size = None
iteration_count = 1024*1000
training_episode_count = 10000
#eval_episode_count = 10
maximum_node_count = 12
maximum_total_credentials = 10
model_path = "toyctf"
render=True
winning_reward=500
save_count = 100

gym_env = cast(CyberBattleEnv, gym.make("CyberBattleToyCtf-v0",winning_reward=winning_reward,
                                        attacker_goal=AttackerGoal(
                                            own_atleast=6
                                        )))

ep = w.EnvironmentBounds.of_identifiers(
    maximum_node_count=maximum_node_count,
    maximum_total_credentials=maximum_total_credentials,
    identifiers=gym_env.identifiers
)

debugging = False
if debugging:
    print(f"port_count = {ep.port_count}, property_count = {ep.property_count}")

    gym_env.environment
    # training_env.environment.plot_environment_graph()
    gym_env.environment.network.nodes
    gym_env.action_space
    gym_env.action_space.sample()
    gym_env.observation_space.sample()
    o0 = gym_env.reset()
    o_test, r, d, i = gym_env.step(gym_env.sample_valid_action())
    o0 = gym_env.reset()

    o0.keys()

    fe_example = w.RavelEncoding(ep, [w.Feature_active_node_properties(ep), w.Feature_discovered_node_count(ep)])
    a = w.StateAugmentation(o0)
    w.Feature_discovered_ports(ep).get(a, None)
    fe_example.encode_at(a, 0)

ddql_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=ddqla.DuelingDQNLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    ),
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=render,
    title="DuelingDQN_0.99_baseline",
    env_str="toyctf",
    save_count = save_count
)

dql_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    ),
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=render,
    title="DQN_0.99_baseline",
    env_str="toyctf",
    save_count = save_count
)

tabularq_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=tqa.QTabularLearner(
        ep=ep,
        gamma=0.015,
        learning_rate=0.01,
        exploit_percentile=100
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
    ),
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=render,
    title="Q-Learning_0.99_baseline",
    env_str="toyctf",
    save_count = save_count
)

ddql_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=ddqla.DuelingDQNLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    ),
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=render,
    title="DuelingDQN_0.015_baseline",
    env_str="toyctf",
    save_count = save_count
)

dql_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    ),
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=render,
    title="DQN_0.015_baseline",
    env_str="toyctf",
    save_count = save_count
)

tabularq_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=tqa.QTabularLearner(
        ep=ep,
        gamma=0.015,
        learning_rate=0.01,
        exploit_percentile=100
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
    ),
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=render,
    title="Q-Learning_0.015_baseline",
    env_str="toyctf",
    save_count = save_count
)

model_str = tabularq_run['model_path']
joblib.dump(tabularq_run['learner'], model_str)

contenders = [
    dql_run,
    tabularq_run
    #deep_drqn_run
]

p.plot_episodes_reward(contenders)
p.plot_episodes_length_each(contenders)
p.plot_episodes_length(contenders)
p.plot_averaged_cummulative_rewards(
    title=f'cummulative reward of all trained model',
    all_runs=contenders)

