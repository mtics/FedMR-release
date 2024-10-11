import copy

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from utils.federated.trainer import FederatedTrainer


class FedAvg(GeneralRecommender):

    def __init__(self, config, dataloader):
        super(FedAvg, self).__init__(config, dataloader)

        self.embed_size = config['latent_size']

        self.item_commonality = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embed_size)

        self.affine_output = torch.nn.Linear(in_features=self.embed_size, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self.apply(xavier_normal_initialization)

    def setItemCommonality(self, item_commonality):
        self.item_commonality = copy.deepcopy(item_commonality)
        # self.item_commonality.freeze = True

    def forward(self, item_indices):
        item_commonality = self.item_commonality(item_indices)

        pred = self.affine_output(item_commonality)
        rating = self.logistic(pred)

        return rating

    def full_sort_predict(self, interaction, *args, **kwargs):
        users = interaction[0]
        items = torch.arange(self.n_items).to(self.device)
        scores = self.forward(items)

        return scores.view(-1)


class FedAvgTrainer(FederatedTrainer):

    def __init__(self, config, model, mg=False):
        super(FedAvgTrainer, self).__init__(config, model, mg)

        self.lr_network = self.config['lr']
        self.lr_args = self.config['lr']

        self.item_commonality = copy.deepcopy(model.item_commonality)

        self.crit, self.independency, self.reg = nn.BCELoss(), nn.MSELoss(), nn.L1Loss()

    def _set_optimizer(self, model):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        param_list = [
            {'params': model.affine_output.parameters(), 'lr': self.lr_network},
            {'params': model.item_commonality.parameters(), 'lr': self.lr_args},
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

    def _set_client(self, *args, **kwargs):
        user, iteration = args

        client_model = copy.deepcopy(self.model)
        client_model.setItemCommonality(self.item_commonality)

        if iteration != 0 and user in self.client_models.keys():
            for key in self.client_models[user].keys():
                client_model.state_dict()[key] = copy.deepcopy(self.client_models[user][key])
            client_model.setItemCommonality(self.item_commonality)

        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        batch_data = batch.to(self.device)
        user, poss, negs = batch_data[0], batch_data[1], batch_data[2]

        self.weights[user[0].item()] += len(poss) + len(negs)

        # construct ratings according to the interaction of positive and negative items
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32)
        ratings[:poss.size(0)] = 1
        ratings = ratings.to(self.device)

        model, optimizer = args[0], args[1]

        optimizer.zero_grad()
        pred = model(items)
        loss = self.calculate_loss(pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

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
            w = self.weights[user] / self.model.n_items

            if i == 0:
                self.item_commonality.weight.data = w * participant_params[user]['item_commonality.weight'].data
            else:
                self.item_commonality.weight.data += w * participant_params[user]['item_commonality.weight'].data
            i += 1

    def _update_hyperparams(self, *args, **kwargs):
        pass

    def calculate_loss(self, interaction, *args, **kwargs):
        pred, truth = interaction, args[0]

        loss = self.crit(pred, truth)

        return loss
