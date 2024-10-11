import math
import random

import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.vae import Encoder, Decoder
from utils.trainer import FederatedTrainer
from utils import evaluation


class FedRAP(torch.nn.Module):
    def __init__(self, args):
        super(FedRAP, self).__init__()
        self.args = args
        self.num_items = args.num_items
        self.latent_dim = args.latent_dim

        self.item_personality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.item_commonality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def setItemCommonality(self, item_commonality):
        self.item_commonality = copy.deepcopy(item_commonality)
        # self.item_commonality.freeze = True

    def forward(self, item_indices):
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        pred = self.affine_output(item_personality + item_commonality)
        rating = self.logistic(pred)

        return rating, item_personality, item_commonality


class FedRAPTrainer(FederatedTrainer):
    """Engine for training & evaluating GMF model"""

    def __init__(self, args):
        self.model = FedRAP(args)

        super(FedRAPTrainer, self).__init__(args)
        print(self.model)

        self.beta, self.gamma = self.args.beta, self.args.gamma

        self.lr_network = self.args.lr
        self.lr_args = self.args.lr * self.model.num_items

        self.item_commonality = copy.deepcopy(self.model.item_commonality)

        self.crit, self.independency, self.reg = nn.BCELoss(), nn.MSELoss(), nn.L1Loss()

    def set_optimizer(self, client_model):
        return torch.optim.SGD([
            {'params': client_model.affine_output.parameters(), 'lr': self.lr_network},
            {'params': client_model.item_personality.parameters(), 'lr': self.lr_args},
            {'params': client_model.item_embed.parameters(), 'lr': self.lr_args},
        ],
            weight_decay=self.args.l2_reg
        )

    def _set_client(self, *args, **kwargs):
        user, iteration = args

        client_model = copy.deepcopy(self.model)
        client_model.setItemCommonality(self.item_commonality)

        if iteration != 0:
            if user in self.client_models.keys():
                for key in self.client_models[user].keys():
                    client_model.state_dict()[key] = copy.deepcopy(self.client_models[user][key])
                # client_model.load_state_dict(self.client_models[user])
                client_model.state_dict()['item_commonality.weight'].data = copy.deepcopy(
                    self.item_commonality.weight.data)

        client_optimizer = self.set_optimizer(client_model)

        return self.fabric.setup(client_model, client_optimizer)

    def _train_one_batch(self, batch, *args, **kwargs):
        batch_data = self.fabric.to_device(batch)
        _, items, ratings = batch_data
        ratings = ratings.float()

        model, optimizer = args[0], args[1]

        optimizer.zero_grad()
        pred, item_personality, item_commonality = model(items)
        loss = self.calculate_loss(pred.view(-1), ratings, item_personality, item_commonality)
        self.fabric.backward(loss)
        optimizer.step()

        return model, optimizer, loss.item()

    def calculate_loss(self, *args, **kwargs):

        pred, truth, item_personality, item_commonality = args[0], args[1], args[2], args[3]
        # pred = pred.unsqueeze(0)

        dummy_target = torch.zeros_like(item_commonality)

        loss = self.crit(pred, truth) \
               - self.beta * self.independency(item_personality, item_commonality) \
               + self.gamma * self.reg(item_commonality, dummy_target)

        return loss

    def _store_client_model(self, *args, **kwargs):
        user, client_model = args

        tmp_dict = copy.deepcopy(client_model.to('cpu').state_dict())
        self.client_models[user] = copy.deepcopy(tmp_dict)
        for key in tmp_dict.keys():
            if 'item_commonality' in key:
                del self.client_models[user][key]

        upload_params = copy.deepcopy(tmp_dict)
        for key in self.client_models[user].keys():
            if 'item_commonality' not in key:
                del upload_params[key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        participant_params = args[0]

        i = 0
        for user in participant_params.keys():
            if i == 0:
                self.item_commonality.weight.data = participant_params[user]['item_commonality.weight'].data
            else:
                self.item_commonality.weight.data += participant_params[user]['item_commonality.weight'].data
            i += 1

        self.item_commonality.weight.data /= len(participant_params)

    def _update_hyperparams(self, *args, **kwargs):
        iteration = args[0]

        self.lr_args *= self.args.decay_rate
        self.lr_network *= self.args.decay_rate

        self.beta = math.tanh(iteration / 10) * self.beta
        self.gamma = math.tanh(iteration / 10) * self.gamma

    @torch.no_grad()
    def evaluate(self, eval_data):
        metrics = None
        all_items = self.fabric.to_device(torch.arange(self.args.n_items))

        for user, loader in eval_data.items():
            client_model, client_optimizer = self._set_client(user, 1)

            client_model.eval()

            client_metrics = None
            for idx, batch in enumerate(loader):
                batch = self.fabric.to_device(batch)

                _, items, train_items = batch

                pred, _, _ = client_model(all_items)
                pred[train_items] = -float('Inf')

                truth = torch.zeros_like(pred)
                truth[items] = 1

                if client_metrics == None:
                    client_metrics = evaluation.recalls_and_ndcgs_for_ks(pred.T, truth.T)
                else:
                    values = evaluation.recalls_and_ndcgs_for_ks(pred.T, truth.T)
                    for key in client_metrics.keys():
                        client_metrics[key] += values[key]

            for key in client_metrics.keys():
                client_metrics[key] = client_metrics[key] / len(loader)

            # print(client_metrics)

            if metrics == None:
                metrics = client_metrics
            else:
                for key in client_metrics.keys():
                    metrics[key] += client_metrics[key]

        for key in metrics.keys():
            metrics[key] = metrics[key] / len(eval_data)

        return metrics['Recall@100'], metrics['NDCG@100']
