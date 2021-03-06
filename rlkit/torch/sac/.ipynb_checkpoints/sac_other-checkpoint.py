from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            env_eval,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            encoder_tau=0.005,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            env_eval=env_eval,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.kl_lambda = kl_lambda
        self.encoder_tau = encoder_tau
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.encoder_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )
        self.curl_optimizer = optimizer_class(
            self.agent.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]
# agent.networks: [self.context_encoder, self.policy]
    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def sample_data(self, indices, encoder=False):
        ''' sample data from replay buffers to construct a training meta-batch '''
        # collect data from multiple tasks for the meta-batch
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            if encoder:
                batch = ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent))
            else:
                batch = ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size))
            o = batch['observations'][None, ...]
            a = batch['actions'][None, ...]
            if encoder and self.sparse_rewards:
                # in sparse reward settings, only the encoder is trained with sparse reward
                r = batch['sparse_rewards'][None, ...]
            else:
                r = batch['rewards'][None, ...]
            no = batch['next_observations'][None, ...]
            t = batch['terminals'][None, ...]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            terms.append(t)
        obs = torch.cat(obs, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_obs = torch.cat(next_obs, dim=0)
        terms = torch.cat(terms, dim=0)
        return [obs, actions, rewards, next_obs, terms]

    def prepare_encoder_data(self, obs, act, rewards, obs_):
        ''' prepare context for encoding '''
        # for now we embed only observations and rewards
        # assume obs and rewards are (task, batch, feat)
        task_data = torch.cat([obs, act, rewards, obs_], dim=2)
        return task_data

    def prepare_context(self, idx):
        ''' sample context from replay buffer and prepare it '''
        batch = ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent))
        obs = batch['observations'][None, ...]
        act = batch['actions'][None, ...]
        rewards = batch['rewards'][None, ...]
        obs_ = batch['next_observations'][None, ...]
        context = self.prepare_encoder_data(obs, act, rewards, obs_)
        return context

    ##### Training #####
    def pretrain(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        batch = self.sample_data(indices, encoder=True)
        batch_ = self.sample_data(indices, encoder=True)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))
        #self.agent2.clear_z(num_tasks=len(indices))
        #self.agent3.clear_z(num_tasks=len(indices))
        for i in range(num_updates):
            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in
                          batch]  # split a batch into several minibatch and update recursively
            mini_batch_ = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in
                           batch_]
            obs_enc, act_enc, rewards_enc, nobs_enc, _ = mini_batch
            obs_enc_, act_enc_, rewards_enc_, nobs_enc_, _ = mini_batch_

            context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc, nobs_enc)
            context_ = self.prepare_encoder_data(obs_enc_, act_enc_, rewards_enc_, nobs_enc_)
            #context2 = self.prepare_encoder_data1(obs_enc, embed_enc, act_enc, rewards_enc, nobs_enc)
            self._pre_take_step(indices, context, context_)

            # stop backprop
            self.agent.detach_z()

    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        batch = self.sample_data(indices, encoder=True)
        batch_ = self.sample_data(indices, encoder=True)
        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        for i in range(num_updates):
            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in batch]#split a batch into several minibatch and update recursively
            mini_batch_ = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in
                           batch_]
            obs_enc, act_enc, rewards_enc, nobs_enc, _ = mini_batch
            obs_enc_, act_enc_, rewards_enc_, nobs_enc_, _ = mini_batch_
            context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc, nobs_enc)
            context_ = self.prepare_encoder_data(obs_enc_, act_enc_, rewards_enc_, nobs_enc_)

            self._take_step(indices, context, context_)

            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)
    def _pre_take_step(self, indices, context1, context1_):
        #num_tasks = len(indices)

        # data is (task, batch, feat)
        
        #positive sample

        self.curl_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda/1000 * kl_div
            kl_loss.backward(retain_graph=True)
        loss.backward()
        self.curl_optimizer.step()
        self.encoder_optimizer.step()

        ptu.soft_update_from_to(
            self.agent.context_encoder, self.agent.context_encoder_target, self.encoder_tau
        )
    def _take_step(self, indices, context1, context1_):
        
        
        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)
        t, b, _ = obs.size()
        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context1)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        policy_mean = policy_mean.view(t,b,-1)
        policy_log_std = policy_log_std.view(t,b,-1)
        
        #positive samples
        policy_outputs_1, task_z_1 = self.agent(obs, context1_)
        new_actions_1, policy_mean_1, policy_log_std_1, log_pi_1 = policy_outputs_1[:4]
        policy_mean_1 = policy_mean_1.view(t,b,-1)#(task, batch, feat)
        policy_log_std_1 = policy_log_std_1.view(t,b,-1)#(task, batch, feat)
        
        #computer policy kl divergence
        policy1 = [[torch.distributions.Normal(mu, torch.exp(log_std)) for mu, log_std in zip(torch.unbind(policy_mean[i]), torch.unbind(policy_log_std[i]))] for i in range(t)]
        policy2 = [[torch.distributions.Normal(mu, torch.exp(log_std)) for mu, log_std in zip(torch.unbind(policy_mean_1[i]), torch.unbind(policy_log_std_1[i]))] for i in range(t)]
        kl_divs = [[torch.distributions.kl.kl_divergence(policy1[i][j], policy2[i][j]) for j in range(b)] for i in range(t)]
        kl_div_sum = torch.mean(torch.stack(kl_divs))
        kl_divs_final = []
        for i in range(len(kl_divs)):
            kl_divs_final.append(torch.mean(torch.stack(kl_divs[i])))
        kl_policy_final = torch.mean(torch.stack(kl_divs_final))
        kl_policy_loss = 1 * kl_policy_final

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        kl_policy_loss.backward(retain_graph=True)
        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.encoder_optimizer.step()
        
        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da

        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss
        #self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                #self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                #self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
            self.eval_statistics['Contrastive Loss'] = np.mean(ptu.get_numpy(loss))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            w=self.agent.state_dict(),
        )
        return snapshot
