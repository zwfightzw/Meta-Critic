import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import Critic_Network, Hot_Plug
from hparams import HyperParams as hp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Meta-Critic of Deep Deterministic Policy Gradients (DDPG_MC)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x1 = F.relu(self.l2(x))
        x2 = self.max_action * torch.tanh(self.l3(x1))
        return x2, x1

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class DDPG_MC_sa(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.feature_critic =  Critic_Network(300+state_dim+action_dim).to(device)
        self.omega_optim = torch.optim.Adam(self.feature_critic.parameters(), lr=hp.aux_lr,
                                            weight_decay=hp.weight_decay)
        feature_net = nn.Sequential(*list(self.actor.children())[:-2])
        self.hotplug = Hot_Plug(feature_net)

        self.max_action = max_action
        self.loss_store = []

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action,_= self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Sample replay buffer for meta test
            x_val, _, _, _, _ = replay_buffer.sample(batch_size)
            state_val = torch.FloatTensor(x_val).to(device)

            # Compute the target Q value
            next_a,_ = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_a)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            a,feature_output = self.actor(state)
            actor_loss = -self.critic(state, a).mean()

            # Delayed policy updates
            if it % 1 == 0:
                concat_output = torch.cat([feature_output, state, action], 1)
                loss_auxiliary = hp.beta * self.feature_critic(concat_output)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                # Part1 of Meta-test stage
                actor_loss.backward(retain_graph=True)
                self.hotplug.update(hp.actor_feature_lr)
                action_val,_ = self.actor(state_val)
                policy_loss_val = self.critic(state_val, action_val)
                policy_loss_val = -policy_loss_val.mean()
                policy_loss_val = policy_loss_val

                # Part2 of Meta-test stage
                loss_auxiliary.backward(create_graph=True)
                self.hotplug.update(hp.actor_feature_lr)
                action_val_new,_ = self.actor(state_val)
                policy_loss_val_new = self.critic(state_val, action_val_new)
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
                self.hotplug.restore()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Store the loss information
                tmp_loss = []
                tmp_loss.append(critic_loss.item())
                tmp_loss.append(actor_loss.item())
                tmp_loss.append(loss_auxiliary.item())
                tmp_loss.append(loss_meta.item())
                self.loss_store.append(tmp_loss)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.feature_critic.state_dict(), '%s/%s_omega.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
