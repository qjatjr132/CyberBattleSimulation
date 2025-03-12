# Cyber Battle Simulation

# Project Documentation

## Start train the model
1. 가상 환경 설정
2. 터미널에서 `toyctf_python_6.py` 파일을 사용하여 훈련 시작

## File Information

### `agent_DuelingDQN.py`, `agent_drqn.py`:
- Dueling DQN 및 DRQN 모델 추가

### `plotting.py`:
- 결과를 시각화하기 위해 'plot_episodes_reward', 'plot_episodes_reward_each', 'plot_episodes_length_each' 3가지 함수를 추가

### `learner.py`:
1. 사용자가 별도로 경로를 설정할 필요 없이 자동으로 경로가 생성
2. `epsilon_greedy_search` 함수를 통해 다음과 같은 다수의 추가적인 인자 받을 수 있음:
   - `start_epi`: 탐험(exploration)을 시작하는 에피소드의 수를 지정
   - `save_count`: 랜더링(rendering)을 저장하는 빈도를 설정
   - `iteration_info_save`: iteration 정보 저장 여부를 결정
   - `trained`: 기존의 모델을 불러와 이어서 학습시킬 때 탐험(exploration)을 제외 (default:False)
   - `trained_new`: 처음 학습 시 탐험(exploration)을 포함 (default:True)
3. 성공률(success rate)을 측정하기 위해 유효한 행동(valid action)과 무효한 행동(invalid action)의 수를 저장
4. 행동(action)에 대한 보상(reward) 구조를 수정할 수 있는 코드 추가 
5. 단기간(적은 timestep) 내에서 종료될 경우, 얻는 보상(reward) 비율을 높이도록 하는 코드 추가 (winning reward 변경)
6. 정상적인 학습을 위해 사용자가 탐험 에피소드(exploration episode)를 설정 가능 (사용자 지정 가능)
7. 유효한 행동(valid action) 상황에서의 시각화가 가능해졌으며, 이때 'render:'로 수정하면 전체를 시각화 가능
8. 유효한 행동들(valid action) 저장
9. 각 에피소드(episode)에 대한 종료 시점(timestep)과 그 때의 보상(reward)을 저장

### `cyberbattle_env.py`:
- 조기 종료 조건을 더욱 세밀하게 조정하기 위해 `__attacker_goal_reached`를 개선함 (환경이 목표에 도달했을 때 더 정확한 탐지와 적절한 조치 가능)
- `throws_on_invalid_actions`을 False로 설정함으로써, 유효하지 않은 행동에 대한 패널티 보상 계산 가능 (잘못된 행동에 대한 더 명확한 피드백 제공 가능)

## 프로그램 특징
- **사이버 공방 시뮬레이터**: Attacker와 Defender 간의 상호작용이 가능하게 구현되어 있어, 복잡한 사이버 보안 상황을 모의실험 가능
- **다양한 노드 구성**: Client, Website, GithubProject, AzureStorage와 같은 다양한 노드가 시뮬레이션 환경에 구현되어 있어, 현실 세계의 다양한 네트워크 환경을 반영
- **오픈 소스 공격 툴킷**: 네트워크 시뮬레이션을 통해, 네트워크가 적의 공격에 어떻게 대처하는지 관찰하고 분석할 수 있는 도구를 제공
- **다양한 시나리오 제공**: Tiny, ToyCTF, Chain Scenario 등 다양한 상황과 환경에서의 시뮬레이션을 제공
- **End Condition 규정**: 다양한 조건에 따라 시뮬레이션의 종료 조건이 설정되어 있어, 실험의 다양성을 확보
- **Reward System**: Positive Reward, Negative Reward, Total Reward, Penalty Reward 등 다양한 보상 체계를 통해 Agent의 학습을 촉진

## 주요 기능
- **Agent와 환경의 상호작용**: Agent의 출력값을 gym action으로 변환시키며, env와 상호작용하게 됩니다. 이를 통해 학습과 실험을 진행 가능
- **공격자의 다양한 Action**: 로컬 취약점(local vulnerability), 원격 취약점(remote vulnerability), 연결 작업(connect actions) 등 공격자가 취할 수 있는 다양한 action이 구현되어 있음
- **학습 가능한 공격자 제공**: 제공된 action space에서 선택한 action의 결과에 따라 결정된 보상을 기반으로 학습하는 공격자를 제공
- **확률적 방어 장치**: 미리 정의된 성공 확률을 기반으로 진행 중인 공격을 탐지하고 완화하는 기본적인 확률적 방어 장치 (basic stochastic defender)를 탑재
- **Agent의 State space 구성**: 마지막 step에서 얻은 전체 환경 정보와 에피소드에서 얻은 특정 노드의 정보를 이용하여 Agent의 State space를 구성
- **시각화 도구**: Attacker의 학습 과정을 시각화하여, 학습 상황을 명확하게 파악 가능
- **Defender Agent의 다양한 Action**: Scan & Reimage, External random event 등 Defender Agent가 취할 수 있는 다양한 action이 구현되어 있음
- **단일 attacker agent 최적화**: 공격자의 효율성을 높이기 위한 최적화 기능이 구현되어 있음

## References
This project has been developed with reference to the following GitHub repository:
- [gym-idsgame](https://github.com/Limmen/gym-idsgame)