import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.trainer import CentralTrainer
from utils.vae import DAE
from utils import evaluation


class CentMAETrainer(CentralTrainer):

    def __init__(self, args):
        self.model = DAE(args)
        super(CentMAETrainer, self).__init__(args)

        print(self.model)

        self.lr = self.args.lr
        self.beta = args.beta
        self.optimizer = self.set_optimizer()

        # Move model to the specified device
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    def set_optimizer(self, *args, **kwargs):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.l2_reg)

    def _train_one_batch(self, batch, *args, **kwargs):
        batch = self.fabric.to_device(batch)

        self.optimizer.zero_grad()
        pred, mu, logvar, _ = self.model(batch)
        loss = self.calculate_loss(pred, batch, mu, logvar)

        self.fabric.backward(loss)

        self.optimizer.step()

        return loss.item()

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

        self.model.eval()

        metrics = None
        for idx, batch in enumerate(eval_data):
            batch = self.fabric.to_device(batch)

            x, truth = batch
            pred, _, _, _ = self.model(x)
            pred[x == 1] = -float("Inf")

            if metrics == None:
                metrics = evaluation.recalls_and_ndcgs_for_ks(pred, truth)
            else:
                values = evaluation.recalls_and_ndcgs_for_ks(pred, truth)
                for key in metrics.keys():
                    metrics[key] += values[key]

        for key in metrics.keys():
            metrics[key] = metrics[key] / len(eval_data)

        return metrics['Recall@{}'.format(self.args.top_k)], metrics['NDCG@{}'.format(self.args.top_k)]
