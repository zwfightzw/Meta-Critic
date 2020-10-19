# Meta-Critic  Network in RL

PyTorch implementation of our NeurIPS 2020 paper `Online Meta-Critic Learning for Off-Policy Actor-Critic Methods'. This paper is located at http://arxiv.org/abs/2003.05334, and will appear in the forthcoming NeurIPS 2020.

> Wei Zhou, Yiying Li, Yongxin Yang, Huaimin Wang, Timothy M. Hospedales. Online Meta-Critic Learning for Off-Policy Actor-Critic Methods. 
34th Conference on Neural Information Processing Systems, 2020.


If you find Meta-Critic useful in your research, please consider citing:
```
  @inproceedings{Zhou2020NeurIPS,
     Author={Zhou, Wei and Li, Yiying and Yang, Yongxin and Wang, Huaimin and Hospedales, Timothy M},
     Title={Online Meta-Critic Learning for Off-Policy Actor-Critic Methods},
     Booktitle={34th Conference on Neural Information Processing Systems},
     Year={2020}
  }
```

## Introduction
Off-Policy Actor-Critic (Off-PAC) methods have proven successful in a variety of continuous control tasks. Normally, the critic's action-value function is updated using temporal-difference, and the critic in turn provides a loss for the actor that trains it to take actions with higher expected return. In this paper, we introduce a novel and flexible meta-critic that observes the learning process and meta-learns an additional loss for the actor that accelerates and improves actor-critic learning. Compared to the vanilla critic, the meta-critic network is explicitly trained to accelerate the learning process; and compared to existing meta-learning algorithms, meta-critic is rapidly learned online for a single task, rather than slowly over a family of tasks. Crucially, our meta-critic framework is designed for off-policy based learners, which currently provide state-of-the-art reinforcement learning sample efficiency. We demonstrate that online meta-critic learning leads to improvements in avariety of continuous control environments when combined with contemporary Off-PAC methods DDPG, TD3 and the state-of-the-art SAC. 

# Getting Started

## Prerequisites

The environment can be run locally using conda, you need to have [Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent) installed. Also, most of our environments currently require a [MuJoCo](https://www.roboti.us/license.html) license.
```
cd ${Miniconda3_PATH}
bash Miniconda3-latest-Linux-x86_64.sh
```

## Conda Installation

1.  [Download](https://www.roboti.us/index.html) and install MuJoCo 1.31 (used for the environment of rllab) and 1.50 from the MuJoCo website. Moreover, for some experimental needs, you need to install the rllab environment [rllab](https://rllab.readthedocs.io/en/latest/index.html).

2.  We assume that the MuJoCo files are extracted to the default location (`~/.mujoco/mjpro150` and `~/.mujoco/mjpro131`). The version of GYM is '0.14.0' and mujoco_py is '1.50.1.68'.

3.  Copy your MuJoCo license key (mjkey.txt) to ~/.mujoco/mjkey.txt:

4. You need to edit your PYTHONPATH to include the rllab directory. You need to have the zip file for MuJoCo 1.31 and the license file ready.:
```
export PYTHONPATH=path_to_rllab:$PYTHONPATH
./scripts/setup_linux.sh
./scripts/setup_mujoco.sh
```

5.  Create and activate conda environment, install meta-critic to enable command line interface.
```
cd ${Meta_Critic_PATH}
conda env create -f environment.yaml
conda activate meta_critic
```

## Examples
### Training and simulating policy agent of DDPG_MC
1.  Enter the directory of TD3_DDPG_MC

```
cd ${TD3_DDPG_MC_PATH}
```
2.  Different design of auxiliary loss network: hw(pi(s))
```
python main.py --env_name HalfCheetahEnv --method DDPG_MC --max_timesteps=3e6
```

3.  Different design of auxiliary loss network: hw(pi(s),s,a)
```
python main.py --env_name HalfCheetahEnv --method DDPG_MC_sa --max_timesteps=3e6
```

### Training and simulating policy agent of TD3_MC
1.  Enter the directory of TD3_DDPG_MC

```
cd ${TD3_DDPG_MC_PATH}
```
2.  Different design of auxiliary loss network: hw(pi(s))
```
python main.py --env_name HalfCheetahEnv --method TD3_MC --max_timesteps=3e6
```

3.  Different design of auxiliary loss network: hw(pi(s),s,a)
```
python main.py --env_name HalfCheetahEnv --method TD3_MC_sa --max_timesteps=3e6
```

### Training and simulating policy agent of SAC_MC
1.  Enter the directory of SAC_MC

```
cd ${SAC_MC_PATH}
```
2.  Different design of auxiliary loss network: hw(pi(s))
```
python main.py --env_name HalfCheetahEnv --method SAC_MC --num_steps=3e6
```

3.  Different design of auxiliary loss network: hw(pi(s),s,a)
```
python main.py --env_name HalfCheetahEnv --method SAC_MC_sa --num_steps=3e6
```
