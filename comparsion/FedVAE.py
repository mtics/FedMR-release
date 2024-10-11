import math
import random

import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.vae import Encoder, Decoder
from utils.trainer import FederatedTrainer
from utils import evaluation, utils

from .MultiVAE import MultiVAE


class FedVAETrainer(FederatedTrainer):
    """Engine for training & evaluating GMF model"""

    def __init__(self, args):
        self.model = MultiVAE(args)
        super(FedVAETrainer, self).__init__(args)
        print(self.model)

        self.beta = args.beta
        self.lr = args.lr

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.args.l2_reg)

        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.decay_rate)

    def _train_one_iter(self, train_loader, iteration):
        # Randomly select a subset of clients
        sampled_clients = utils.sampleClients(
            list(train_loader.keys()), self.args.clients_sample_strategy,
            self.args.clients_sample_ratio, self.last_participants
        )

        # Store the selected clients for the next round
        self.last_participants = sampled_clients

        participant_params = {}
        train_loss = 0

        self.model.train()
        self.optimizer.zero_grad()
        for user in sampled_clients:
            client_loader = train_loader[user]
            client_model = copy.deepcopy(self.model)
            client_loss = 0
            for idx, batch in enumerate(client_loader):
                x = self.fabric.to_device(batch)

                pred, mu, logvar, _ = client_model(x)
                loss = self.calculate_loss(pred, x, mu, logvar)
                self.fabric.backward(loss)

                for pA, pB in zip(client_model.parameters(), self.model.parameters()):
                    pB.grad = pA.grad + (pB.grad if pB.grad is not None else 0)

                client_loss += loss.item()

            train_loss += client_loss / len(client_loader)

        for p in self.model.parameters():
            p.grad /= len(sampled_clients)

        self.optimizer.step()
        self.scheduler.step()

        self._update_hyperparams(iteration)

        return train_loss / len(sampled_clients)

    def calculate_loss(self, *args, **kwargs):
        pred, truth, mu, logvar = args[0], args[1], args[2], args[3]

        # BCE = -torch.mean(torch.sum(F.log_softmax(pred, 0) * truth, -1))
        BCE = -(F.log_softmax(pred, 1) * truth).sum(1).mean()

        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        loss = BCE + self.beta * KLD

        return loss

    def _update_hyperparams(self, *args, **kwargs):
        iteration = args[0]

        self.beta = math.tanh(iteration / 10) * self.beta
        self.lr *= self.args.decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    @torch.no_grad()
    def evaluate(self, eval_data):

        all_pred, all_truth = [], []

        self.model.eval()
        for user, loader in eval_data.items():
            client_model = copy.deepcopy(self.model)

            batch = list(eval_data[user])[0]
            batch = self.fabric.to_device(batch)
            x, truth = batch

            pred, _, _, _ = client_model(x)
            pred[x == 1] = -float("Inf")

            all_pred.append(pred)
            all_truth.append(truth)

        tensor_pred = torch.cat(all_pred, dim=0)
        tensor_truth = torch.cat(all_truth, dim=0)

        metrics = evaluation.recalls_and_ndcgs_for_ks(tensor_pred, tensor_truth)

        return metrics['Recall@{}'.format(self.args.top_k)], metrics['NDCG@{}'.format(self.args.top_k)]
