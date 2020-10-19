import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import Critic_Network, Hot_Plug
from hparams import HyperParams as hp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Meta-Critic of Twin Delayed Deep Deterministic Policy Gradients (TD3_MC)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Actor_Feature(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor_Feature, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

class TD3_MC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)

        self.actor_feature = Actor_Feature(state_dim, action_dim, max_action).to(device)
        self.actor_feature_target = Actor_Feature(state_dim, action_dim, max_action).to(device)
        self.actor_feature_target.load_state_dict(self.actor_feature.state_dict())
        self.actor_feature_optimizer = torch.optim.Adam(self.actor_feature.parameters(), lr=hp.actor_feature_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.feature_critic = Critic_Network(300).to(device)
        self.omega_optim = torch.optim.Adam(self.feature_critic.parameters(), lr=hp.aux_lr, weight_decay=hp.weight_decay)
        feature_net = nn.Sequential(*list(self.actor_feature.children())[:-1])
        self.hotplug = Hot_Plug(feature_net)

        self.max_action = max_action
        self.loss_store = []

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state_feature = self.actor_feature(state)
        return self.actor(state_feature).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Sample replay buffer for meta train
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Sample replay buffer for meta test
            x_val, _, _, _, _ = replay_buffer.sample(batch_size)
            state_val = torch.FloatTensor(x_val).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_state_feature = self.actor_feature_target(next_state)
            next_action = (self.actor_target(next_state_feature) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                state_feature = self.actor_feature(state)
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state_feature)).mean()
                loss_auxiliary = hp.beta * self.feature_critic(state_feature)

                self.actor_optimizer.zero_grad()
                self.actor_feature_optimizer.zero_grad()
                # Part1 of Meta-test stage
                actor_loss.backward(retain_graph=True)
                self.hotplug.update(hp.actor_feature_lr)
                state_feature_val = self.actor_feature(state_val)
                policy_loss_val = self.critic.Q1(state_val, self.actor(state_feature_val))
                policy_loss_val = -policy_loss_val.mean()
                policy_loss_val = policy_loss_val

                # Part2 of Meta-test stage
                loss_auxiliary.backward(create_graph=True)
                self.hotplug.update(hp.actor_feature_lr)
                state_feature_val_new = self.actor_feature(state_val)
                policy_loss_val_new = self.critic.Q1(state_val, self.actor(state_feature_val_new))
                policy_loss_val_new = -policy_loss_val_new.mean()
                policy_loss_val_new = policy_loss_val_new

                utility = policy_loss_val - policy_loss_val_new
                utility = torch.tanh(utility)
                loss_meta = -utility

                # Meta optimization of auxilary network
                self.omega_optim.zero_grad()
                grad_omega = torch.autograd.grad(loss_meta, self.feature_critic.parameters())
                for gradient, variable in zip(grad_omega, self.feature_critic.parameters()):
                    variable.grad.data = gradient
                self.omega_optim.step()
                self.actor_optimizer.step()
                self.actor_feature_optimizer.step()
                self.hotplug.restore()

                # Store the loss information
                tmp_loss = []
                tmp_loss.append(critic_loss.item())
                tmp_loss.append(actor_loss.item())
                tmp_loss.append(loss_auxiliary.item())
                tmp_loss.append(loss_meta.item())
                self.loss_store.append(tmp_loss)

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor_feature.parameters(), self.actor_feature_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor_feature.state_dict(), '%s/%s_actor_feature.pth' % (directory, filename))
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor_feature.load_state_dict(torch.load('%s/%s_actor_feature.pth' % (directory, filename)))
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
