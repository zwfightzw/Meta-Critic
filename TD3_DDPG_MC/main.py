import numpy as np
import torch
import gym
import argparse
import os
import utils
import DDPG_MC
import DDPG_MC_sa
import TD3_MC
import TD3_MC_sa
import time
from hparams import HyperParams as hp
import dateutil.tz
import datetime

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.ant_env import AntEnv

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps<1001:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            steps +=1

    avg_reward /= eval_episodes

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")
    return avg_reward

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="DDPG_MC")  # Policy name
    parser.add_argument("--env_name", default="HalfCheetahEnv")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=1e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e7, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument('--actor_lr', type=float, default=1e-3)  # learning rate
    parser.add_argument('--critic_lr', type=float, default=1e-3)  # learning rate
    parser.add_argument('--aux_lr', type=float, default=1e-3)  # learning rate
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # weight decay
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--gpu_id", default=0, type=int)  # The id of GPU
    args = parser.parse_args()

    # Create the log directory
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    time_dir = now.strftime('%Y_%m_%d_%H_%M_%S')
    time_dir = "%s_%s_%s" % (args.env_name, str(args.seed), time_dir)
    file_name = time_dir
    if not os.path.exists('logs/%s/%s/' % (args.method, time_dir)):
        os.makedirs('logs/%s/%s/' % (args.method, time_dir))
    if not os.path.exists('model_output/%s/%s/' % (args.method, time_dir)):
        os.makedirs('model_output/%s/%s/' % (args.method, time_dir))

    flags_log = os.path.join('logs/%s/%s/' % (args.method, time_dir), 'log.txt')
    model_path = 'model_output/%s/%s' % (args.method, time_dir)
    plot_path = 'logs/%s/%s' % (args.method,time_dir)

    # Store the parameter of test
    utils.write_log(args, flags_log)
    localtime = time.asctime(time.localtime(time.time()))
    utils.write_log(localtime, flags_log)

    # Build the continuous control task environment
    if args.env_name == 'HalfCheetahEnv':
        env = HalfCheetahEnv()
        env.seed(args.seed)
        env = normalize(env)
    elif args.env_name == 'AntEnv':
        env = AntEnv()
        env.seed(args.seed)
        env = normalize(env)
    else:
        env = gym.make(args.env_name)
        env.seed(args.seed)
    # Set seeds
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.method == "TD3_MC":
        policy = TD3_MC.TD3_MC(state_dim, action_dim, max_action, args)
    elif args.method == "TD3_MC_sa":
        policy = TD3_MC_sa.TD3_MC_sa(state_dim, action_dim, max_action, args)
    elif args.method == "DDPG_MC":
        policy = DDPG_MC.DDPG_MC(state_dim, action_dim, max_action, args)
    elif args.method == "DDPG_MC_sa":
        policy = DDPG_MC_sa.DDPG_MC_sa(state_dim, action_dim, max_action, args)

    utils.write_log(str(policy.feature_critic), flags_log)
    utils.write_log(str(policy.omega_optim), flags_log)
    utils.write_log(str(policy.actor_optimizer), flags_log)
    utils.write_log(str(policy.critic_optimizer), flags_log)
    replay_buffer = utils.ReplayBuffer()
    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_timesteps = 0
    episode_reward = 0

    while total_timesteps < args.max_timesteps:

        if done or episode_timesteps >= 1000:

            if total_timesteps != 0:
                utils.write_log(total_timesteps, flags_log)
                print("Total T %d Episode Num %d Episode T %d Reward %f" % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
                if args.method == 'TD3_MC' or args.method == 'TD3_MC_sa':
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
                                 args.policy_noise, args.noise_clip, args.policy_freq)
                else:
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(policy))

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == 1000 else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))
        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(policy))
    if args.save_models: policy.save("%s" % (file_name), directory=model_path)
    np.save('logs/%s/%s/evaluation_reward' % (args.method, time_dir), evaluations)
    np.save('logs/%s/%s/loss_inf' % (args.method, time_dir), policy.loss_store)
    utils.plot_results(plot_path)
