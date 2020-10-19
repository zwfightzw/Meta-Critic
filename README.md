# Meta-Critic  Network in RL

PyTorch implementation of our NIPS 2020 paper on Online Meta-Critic Learning for Off-Policy Actor-Critic Methods.

Wei Zhou, Yiying Li, Yongxin Yang, Huaimin Wang, Timothy M. Hospedales

Online Meta-Critic Learning for Off-Policy Actor-Critic Methods

Advances in Neural Information Processing Systems 34 (NIPS 2020)

http://arxiv.org/abs/2003.05334

Please cite this paper when using this code for your research.

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
