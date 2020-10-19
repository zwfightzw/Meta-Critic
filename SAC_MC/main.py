import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from SAC_MC import SAC_MC
from SAC_MC_sa import SAC_MC_sa
from replay_memory import ReplayMemory
import os
import datetime
import dateutil.tz
import utils

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.ant_env import AntEnv

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env_name', default="Walker2d-v2",
                    help='name of the environment to run')
parser.add_argument('--policy', default="Gaussian",
                    help='algorithm to use: Gaussian | Deterministic')
parser.add_argument('--method', default="SAC_MC",
                    help='method to use: SAC_MC | SAC_MC_sa')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr_policy', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--lr_critic', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--lr_aux', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight_decay_aux', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Temperature parameter α automaically adjusted.')
parser.add_argument('--seed', type=int, default=2, metavar='N',
                    help='random seed (default: 456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=10000001, metavar='N',
                    help='maximum number of steps (default: 10000001)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--gpu_id', type=int, default=1, metavar='N',
                    help='The id of GPU device')

args = parser.parse_args()

# Set the GPU id
torch.cuda.set_device(args.gpu_id)
# Environment
if args.env_name == 'HalfCheetahEnv':
    env= HalfCheetahEnv()
    env.seed(args.seed)
    env = normalize(env)
elif args.env_name == 'AntEnv':
    env = AntEnv()
    env.seed(args.seed)
    env = normalize(env)
else:
    # gym environments
    env = gym.make(args.env_name)
    env.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
# save results
now = datetime.datetime.now(dateutil.tz.tzlocal())
time_dir = now.strftime('%Y_%m_%d_%H_%M_%s')
file_dir = '%s_%s_%s' % (args.env_name,args.seed,time_dir)
if not os.path.exists('logs/%s/%s/' % (args.policy, file_dir)):
    os.makedirs('logs/%s/%s/' % (args.policy, file_dir))
if not os.path.exists('model_output/%s/%s/' % (args.policy, file_dir)):
    os.makedirs('model_output/%s/%s/' % (args.policy, file_dir))
# Archive log information
flags_log = os.path.join('logs/%s/%s/' % (args.policy, file_dir), 'log.txt')
# Save the parameter of networks
model_path = 'model_output/%s/%s' % (args.policy, file_dir)
# Draw the learning curves in real time
plot_path = 'logs/%s/%s' % (args.policy, file_dir)

# Agent
if args.method =='SAC_MC':
    agent = SAC_MC(env.observation_space.shape[0], env.action_space, args)
if args.method =='SAC_MC_sa':
    agent = SAC_MC_sa(env.observation_space.shape[0], env.action_space, args)

# Restore the algorithm setting
utils.write_log('calculate the average gradient of policy', flags_log)
utils.write_log(args, flags_log)
utils.write_log(agent.policy_optim,flags_log)
utils.write_log(agent.critic_optim,flags_log)
if args.method =='SAC_MC_sa' or args.method =='SAC_MC':
    utils.write_log(agent.omega_optim, flags_log)
else:
    utils.write_log(agent.opt_meta_reg, flags_log)
# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

exploration_reward = []
evaluation_reward = []
update_information = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done and episode_steps < 1000:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks, and restore the loss information
                update_info = agent.update_parameters(memory,args.batch_size,updates)
                update_information.append(update_info)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        mask = 1 if episode_steps == 1000 else float(not done)
        memory.push(state, action, reward, next_state, mask) # Append transition to memory
        state = next_state

        if total_numsteps % 1000 == 0 and args.eval == True:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state = env.reset()
                episode_reward = 0
                steps = 0
                done = False
                while not done and steps < 1001:
                    action = agent.select_action(state, eval=True)
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    steps +=1
                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")
            print(plot_path)
            evaluation_reward.append(round(avg_reward, 2))

    if total_numsteps > args.num_steps:
        break

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    exploration_reward.append(round(episode_reward,2))

np.save('logs/%s/%s/evaluation_reward' % (args.policy, file_dir), evaluation_reward)
np.save('logs/%s/%s/exploration_reward'% (args.policy, file_dir),exploration_reward)
np.save('logs/%s/%s/update_information'% (args.policy, file_dir),update_information)
actor_path = 'model_output/%s/%s/actor' % (args.policy, file_dir)
critic_path = 'model_output/%s/%s/critic' % (args.policy, file_dir)
fc_path = 'model_output/%s/%s/fc' % (args.policy, file_dir)
agent.save_model(actor_path, critic_path, fc_path)

env.close()

