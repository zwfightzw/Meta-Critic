import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update, Hot_Plug, Critic_Network, l1_penalty
from model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC_MC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=3e-4)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr_policy)
            self.policy_lr = args.lr_policy

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr_policy)
            self.policy_lr = args.lr_policy

        self.feature_critic = Critic_Network(args.hidden_size).cuda()
        self.omega_optim = torch.optim.Adam(self.feature_critic.parameters(), lr=args.lr_aux,
                                            weight_decay=args.weight_decay_aux)
        feature_net = nn.Sequential(*list(self.policy.children())[:-2])
        self.hotplug = Hot_Plug(feature_net)

        def get_layer(model):
            count = 0
            para_optim = []
            for k in model.children():
                count += 1
                # 6 should be changed properly
                for param in k.parameters():
                    para_optim.append(param)
            return para_optim
        self.param_optim_theta = get_layer(self.policy)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch_val, *_ = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        state_batch_val = torch.FloatTensor(state_batch_val).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, *_ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _, other_output = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        ###############################################################################
        ##############################    PART 1        ###############################
        ###############################################################################
        ###############################################################################
        loss_auxiliary = self.feature_critic(other_output)

        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.hotplug.update(self.policy_lr)

        pi_val, log_pi_val, *_ = self.policy.sample(state_batch_val)

        qf1_pi_val, qf2_pi_val = self.critic(state_batch_val, pi_val)
        min_qf_pi_val = torch.min(qf1_pi_val, qf2_pi_val)

        policy_loss_val = ((self.alpha * log_pi_val) - min_qf_pi_val).mean()

        ###############################################################################
        ##############################    PART 2        ###############################
        ###############################################################################
        ###############################################################################
        loss_auxiliary.backward(create_graph=True)

        abs_theta = 0.0
        for p in self.param_optim_theta:
            abs_theta += l1_penalty(p._grad.data).item()

        self.hotplug.update(self.policy_lr)

        pi_val_new, log_pi_val_new, *_ = self.policy.sample(state_batch_val)

        qf1_pi_val_new, qf2_pi_val_new = self.critic(state_batch_val, pi_val_new)
        min_qf_pi_val_new = torch.min(qf1_pi_val_new, qf2_pi_val_new)

        policy_loss_val_new = ((self.alpha * log_pi_val_new) - min_qf_pi_val_new).mean()

        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        loss_meta = -utility

        self.omega_optim.zero_grad()
        grad_omega = torch.autograd.grad(loss_meta, self.feature_critic.parameters())
        for gradient, variable in zip(grad_omega, self.feature_critic.parameters()):
            variable.grad.data = gradient
        self.omega_optim.step()

        self.policy_optim.step()
        self.hotplug.restore()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), loss_auxiliary.item(), loss_meta.item(), abs_theta

    # Save model parameters
    def save_model(self, actor_path=None, critic_path=None, fc_path=None):
        print('Saving models to {}, {} and {}'.format(actor_path, critic_path, fc_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.feature_critic.state_dict(), fc_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path, fc_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if fc_path is not None:
            self.feature_critic.load_state_dict(torch.load(fc_path))
            print('success load the fc model')

