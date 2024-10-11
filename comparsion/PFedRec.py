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

        self.item_embedding = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        item_embedding = self.item_embedding(item_indices)

        pred = self.affine_output(item_embedding)
        rating = self.logistic(pred)

        return rating


class PFedRecTrainer(FederatedTrainer):
    """Engine for training & evaluating GMF model"""

    def __init__(self, args):
        self.model = FedRAP(args)

        super(PFedRecTrainer, self).__init__(args)
        print(self.model)

        self.lr_network = self.args.lr
        self.lr_args = self.args.lr * self.model.num_items

        self.crit = nn.BCELoss()

    def set_optimizer(self, client_model):
        return torch.optim.SGD([
            {'params': client_model.affine_output.parameters(), 'lr': self.lr_network},
            {'params': client_model.id_embed.parameters(), 'lr': self.lr_args},
        ],
            weight_decay=self.args.l2_reg
        )

    def _set_client(self, *args, **kwargs):
        user, iteration = args

        client_model = copy.deepcopy(self.model)

        if iteration != 0:
            for key in self.global_model.keys():
                client_model.state_dict()[key].data = self.global_model[key].data

            if user in self.client_models.keys():
                for key in self.client_models[user].keys():
                    client_model.state_dict()[key].data = self.client_models[user][key].data

        # client_model.load_state_dict(self.client_models[user])

        client_optimizer = self.set_optimizer(client_model)

        return self.fabric.setup(client_model, client_optimizer)

    def _train_one_batch(self, batch, *args, **kwargs):
        batch_data = self.fabric.to_device(batch)
        _, items, ratings = batch_data
        ratings = ratings.float()

        model, optimizer = args[0], args[1]

        optimizer.zero_grad()
        pred = model(items)
        loss = self.crit(pred.view(-1), ratings)
        self.fabric.backward(loss)
        optimizer.step()

        return model, optimizer, loss.item()

    def _store_client_model(self, *args, **kwargs):
        user, client_model = args
        self.client_models[user] = copy.deepcopy(client_model.to('cpu').state_dict())
        upload_params = copy.deepcopy(self.client_models[user])
        for key in client_model.state_dict().keys():
            if key != 'affine_output.weight':
                del self.client_models[user][key]

        del upload_params['affine_output.weight']

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        participant_params = args[0]

        i = 0
        for user in participant_params.keys():
            if i == 0:
                self.global_model = copy.deepcopy(participant_params[user])
            else:
                for key in participant_params[user].keys():
                    self.global_model[key] += participant_params[user][key]
            i += 1

        for key in self.global_model.keys():
            self.global_model[key].data /= len(participant_params)

    def _update_hyperparams(self, *args, **kwargs):
        iteration = args[0]

        self.lr_args *= self.args.decay_rate
        self.lr_network *= self.args.decay_rate

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

                pred = client_model(all_items)
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
