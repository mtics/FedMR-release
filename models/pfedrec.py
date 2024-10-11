import copy

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from utils.federated.trainer import FederatedTrainer


class PFedRec(GeneralRecommender):

    def __init__(self, config, dataloader):
        super(PFedRec, self).__init__(config, dataloader)

        self.embed_size = config['embedding_size']
        self.item_embed = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embed_size)
        self.affine_output = torch.nn.Linear(in_features=self.embed_size, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self.apply(xavier_normal_initialization)

    def forward(self, item_indices):
        item_embed = self.item_embed(item_indices)

        pred = self.affine_output(item_embed)
        rating = self.logistic(pred)

        return rating

    def full_sort_predict(self, interaction, *args, **kwargs):
        items = torch.arange(self.n_items).to(self.device)
        scores = self.forward(items)

        return scores.view(-1)


class PFedRecTrainer(FederatedTrainer):

    def __init__(self, config, model, mg=False):
        super(PFedRecTrainer, self).__init__(config, model, mg)

        self.lr_network = self.config['lr']
        self.lr_args = self.config['lr'] * model.n_items

        self.crit = nn.BCELoss()

    def _set_optimizer(self, model):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        param_list = [
            {'params': model.affine_output.parameters(), 'lr': self.lr_network},
            {'params': model.item_embed.parameters(), 'lr': self.lr_args},
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
        if iteration != 0:
            # load the client model
            if user in self.client_models.keys():
                for key in self.client_models[user].keys():
                    client_model.state_dict()[key] = copy.deepcopy(self.client_models[user][key])
            # load the global parameters
            for key in self.global_model.keys():
                client_model.state_dict()[key] = copy.deepcopy(self.global_model[key])

        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        batch_data = batch.to(self.device)
        _, poss, negs = batch_data[0], batch_data[1], batch_data[2]
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
            if 'affine_output' not in key:
                del self.client_models[user][key]

        upload_params = copy.deepcopy(tmp_dict)
        for key in self.client_models[user].keys():
            if 'affine_output' in key:
                del upload_params[key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        participant_params = args[0]

        i = 0
        for user in participant_params.keys():
            if i == 0:
                self.global_model = copy.deepcopy(participant_params[user])
            else:
                for key in participant_params[user].keys():
                    self.global_model[key].data += participant_params[user][key].data
            i += 1

        for key in self.global_model.keys():
            self.global_model[key].data /= len(participant_params)

    def calculate_loss(self, interaction, *args, **kwargs):
        pred, truth = interaction, args[0]

        loss = self.crit(pred, truth)

        return loss
