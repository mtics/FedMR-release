import copy
import math

import torch
from torch import optim

from models.MR.modules import MR
from utils.federated.trainer import FederatedTrainer


class FedMR(MR):
    def __init__(self, config, dataloader):
        super(FedMR, self).__init__(config, dataloader)


class FedMRTrainer(FederatedTrainer):

    def __init__(self, config, model, mg=False):
        super(FedMRTrainer, self).__init__(config, model, mg)

        self.lr = config['lr']

        self.client_lrs = {}

        self.id_embed = copy.deepcopy(model.id_embed)
        self.global_model = copy.deepcopy(model)
        self.optimizer = self._set_optimizer(self.model, self.lr)

        self.crit = torch.nn.BCELoss()

    def _set_client(self, *args, **kwargs):
        user, iteration = args

        client_model = copy.deepcopy(self.model)

        if iteration != 0:
            if user in self.client_models.keys():
                for key in self.client_models[user].keys():
                    client_model.state_dict()[key].data.copy_(self.client_models[user][key].data)
            client_model.fusion = copy.deepcopy(self.global_model.fusion)

        if not self.config['local_id']:
            client_model.set_id_embed(self.id_embed)

        client_model = client_model.to(self.device)

        # 定义优化器
        client_lr = self.client_lrs[user] if user in self.client_lrs.keys() else self.lr
        client_optimizer = self._set_optimizer(client_model, client_lr)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        txt_features = self.t_feat.to(self.device)
        vis_features = self.v_feat.to(self.device)

        batch_data = batch.to(self.device)
        _, poss, negs = batch_data[0], batch_data[1], batch_data[2]

        # construct ratings according to the interaction of positive and negative items
        items = torch.cat([poss, negs], dim=0)
        ratings = torch.cat([
            torch.ones(poss.size(0), dtype=torch.float32),
            torch.zeros(negs.size(0), dtype=torch.float32)
        ], dim=0).to(self.device)

        model, optimizer = args[0], args[1]

        optimizer.zero_grad()
        pred = model(items, txt_features, vis_features)
        loss = self.calculate_loss(pred, ratings)
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        user, client_model = args

        user_dict = client_model.to('cpu').state_dict()
        self.client_models[user] = copy.deepcopy(user_dict)

        upload_params = {
            name: param.grad.clone() for name, param in client_model.named_parameters() if param.grad is not None
        }

        upload_params['id_embed.weight'] = user_dict['id_embed.weight'].clone()

        for key in user_dict.keys():
            if any(sub in key for sub in ['id_embed', 'mlp', 'attention', 'router', 'gate']):
                del self.client_models[user][key]
            else:
                del upload_params[key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        participant_params = args[0]

        self.global_model.train()
        self.optimizer.zero_grad()

        # 初始化全局模型梯度与id_embed权重
        grad_accumulator = {}
        id_embed_weight_sum = None

        for user, params in participant_params.items():
            # 累加梯度
            for name, param in self.global_model.named_parameters():
                if name in params and params[name] is not None:
                    if name not in grad_accumulator:
                        grad_accumulator[name] = params[name].clone()
                    else:
                        grad_accumulator[name] += params[name].clone()

            # 累加id_embed权重
            if 'id_embed.weight' in params:
                if id_embed_weight_sum is None:
                    id_embed_weight_sum = params['id_embed.weight'].data.clone()
                else:
                    id_embed_weight_sum += params['id_embed.weight'].data.clone()

        # 平均梯度与id_embed权重
        num_participants = len(participant_params)
        for name, param in self.global_model.named_parameters():
            if name in grad_accumulator:
                param.grad = grad_accumulator[name] / num_participants
            if 'id_embed' in name:
                if id_embed_weight_sum is not None:
                    param.data = id_embed_weight_sum / num_participants

        self.optimizer.step()

        # Update the id_embed weight
        self.id_embed.weight.data = self.global_model.id_embed.weight.data.clone()

    def _set_optimizer(self, model, lr):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        # 基础的优化器参数列表
        param_list = [
            {'params': model.predictor.parameters(), 'lr': lr},
            {'params': model.id_embed.parameters(), 'lr': lr * self.model.n_items},
            {'params': model.fusion.parameters(), 'lr': lr}
        ]

        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(param_list, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(param_list, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _update_hyperparams(self, *args, **kwargs):
        iteration = args[0]

        self.lr = self.config['lr'] * math.exp(- self.config['decay_rate'] * iteration)

    def calculate_loss(self, *args, **kwargs):
        pred, truth = args[0], args[1]
        # Check whether pred and truth have the same shape
        if pred.shape != truth.shape:
            truth = truth.view(pred.shape)  # reshape truth to match pred

        loss = self.crit(pred, truth)

        return loss
