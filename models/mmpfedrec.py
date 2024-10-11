import copy

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from models.MR.modules import FusionLayer
from utils.federated.trainer import FederatedTrainer


class MMPFedRec(GeneralRecommender):

    def __init__(self, config, dataloader):
        super(MMPFedRec, self).__init__(config, dataloader)

        self.embed_size = config['embedding_size']
        self.latent_size = config['latent_size']

        self.item_embed = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embed_size)

        self.fusion = FusionLayer(self.embed_size, fusion_module=config['fusion_module'],
                                  latent_dim=self.latent_size)

        self.affine_output = torch.nn.Linear(in_features=self.latent_size, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self.apply(xavier_normal_initialization)

    def forward(self, item_indices, txt_embed, vision_embed):
        item_embed = self.item_embed(item_indices)

        txt = txt_embed[item_indices]
        vision = vision_embed[item_indices]

        # construct zero tensor
        dummy_target = torch.zeros_like(txt).to(self.device)

        # out = self.fusion(item_embed, txt, vision)
        out = self.fusion(item_embed, txt, dummy_target)

        pred = self.affine_output(out)
        rating = self.logistic(pred)

        return rating

    def full_sort_predict(self, interaction, *args, **kwargs):
        txt_embed, vis_embed = args[0], args[1]

        items = torch.arange(self.n_items).to(self.device)

        scores = self.forward(items, txt_embed, vis_embed)

        return scores.view(-1)


class MMPFedRecTrainer(FederatedTrainer):

    def __init__(self, config, model, mg=False):
        super(MMPFedRecTrainer, self).__init__(config, model, mg)

        self.lr_network = self.config['lr']
        self.lr_args = self.config['lr'] * model.n_items

        self.fusion = copy.deepcopy(model.fusion)
        self.optimizers = torch.optim.Adam(self.fusion.parameters(), lr=self.lr_network)

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

            client_model.fusion = copy.deepcopy(self.fusion)

        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        txt_features = self.t_feat.to(self.device)
        vis_features = self.v_feat.to(self.device)

        batch_data = batch.to(self.device)
        _, poss, negs = batch_data[0], batch_data[1], batch_data[2]
        # construct ratings according to the interaction of positive and negative items
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32)
        ratings[:poss.size(0)] = 1
        ratings = ratings.to(self.device)

        model, optimizer = args[0], args[1]

        optimizer.zero_grad()
        pred = model(items, txt_features, vis_features)
        loss = self.calculate_loss(pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        user, client_model = args

        user_dict = copy.deepcopy(client_model.to('cpu').state_dict())
        self.client_models[user] = copy.deepcopy(user_dict)
        for key in user_dict.keys():
            if 'affine_output' not in key:
                del self.client_models[user][key]

        upload_params = {
            name: param.grad.clone() for name, param in client_model.fusion.named_parameters() if param.grad is not None
        }
        for key in self.client_models[user].keys():
            if any(sub in key for sub in ['mlp', 'attention', 'gate']):
                del self.client_models[user][key]
            else:
                if key in upload_params.keys():
                    del upload_params[key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        participant_params = args[0]

        self.fusion.train()
        self.optimizer.zero_grad()

        grad_accumulator = {}
        i = 0
        for user, params in participant_params.items():

            if i == 0:
                self.global_model = copy.deepcopy(participant_params[user])
            else:
                for key in participant_params[user].keys():
                    self.global_model[key].data += participant_params[user][key].data
            i += 1

            # 累加梯度
            for name, param in self.fusion.named_parameters():
                if name in params and params[name] is not None:
                    if name not in grad_accumulator:
                        grad_accumulator[name] = params[name].clone()
                    else:
                        grad_accumulator[name] += params[name].clone()

                    # 平均梯度与id_embed权重
                num_participants = len(participant_params)
                for name, param in self.fusion.named_parameters():
                    if name in grad_accumulator:
                        param.grad = grad_accumulator[name].to(param.device) / num_participants

                self.optimizer.step()

        for key in self.global_model.keys():
            self.global_model[key].data /= len(participant_params)

    def calculate_loss(self, interaction, *args, **kwargs):
        pred, truth = interaction, args[0]

        loss = self.crit(pred, truth)

        return loss
