import copy

import torch
import torch.nn as nn
from torch import optim

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from models.MR.modules import FusionLayer
from utils.federated.trainer import FederatedTrainer


class GMF(nn.Module):
    def __init__(self, n_users, n_items, latent_dim):
        super(GMF, self).__init__()
        self.num_users = n_users
        self.num_items = n_items
        self.latent_dim = latent_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating


class MLP(nn.Module):
    def __init__(self, n_users, n_items, latent_dim, layers):
        super(MLP, self).__init__()

        self.num_users = n_users
        self.num_items = n_items
        self.latent_dim = latent_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=layers[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating


class MMFedNCF(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(MMFedNCF, self).__init__(config, dataloader)

        self.embed_size = config['embedding_size']
        self.latent_size = config['latent_size']

        self.latent_dim_mf = self.latent_size
        self.latent_dim_mlp = self.latent_size

        self.item_commonality = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embed_size)

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim_mf)

        self.fusion = FusionLayer(self.embed_size, fusion_module=config['fusion_module'],
                                  latent_dim=config['latent_size'])

        layers = [2 * self.latent_dim_mlp, self.latent_dim_mlp, self.latent_dim_mlp // 2, self.latent_dim_mlp // 4]

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=layers[-1] + self.latent_dim_mf, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        self.apply(xavier_normal_initialization)

    def setItemCommonality(self, item_commonality):
        self.item_commonality = copy.deepcopy(item_commonality)
        # self.item_commonality.freeze = True

    def forward(self, user_indices, item_indices, txt_embed, vision_embed):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        txt = txt_embed[item_indices]
        vision = vision_embed[item_indices]
        item_commonality = self.item_commonality(item_indices)

        # construct zero tensor
        dummy_target = torch.zeros_like(txt).to(self.device)

        # out = self.fusion(item_commonality, txt, vision)
        out = self.fusion(item_commonality, txt, dummy_target)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp + out], dim=-1)  # the concat latent vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf + out)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def full_sort_predict(self, interaction, *args, **kwargs):
        txt_embed, vis_embed = args[0], args[1]

        user = interaction[0, 0]
        users = torch.full((self.n_items,), user, dtype=torch.long).to(self.device)

        items = torch.arange(self.n_items).to(self.device)
        scores = self.forward(users, items, txt_embed, vis_embed)

        return scores.view(-1)


class MMFedNCFTrainer(FederatedTrainer):

    def __init__(self, config, model, mg=False):
        super(MMFedNCFTrainer, self).__init__(config, model, mg)

        self.lr = self.config['lr']

        self.item_commonality = copy.deepcopy(model.item_commonality)

        self.fusion = copy.deepcopy(model.fusion)
        self.optimizer = optim.Adam(self.fusion.parameters(), lr=self.lr)

        self.crit, self.mae = nn.BCELoss(), nn.L1Loss()

    def _set_optimizer(self, model):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        param_list = [
            {'params': model.parameters(), 'lr': self.lr},
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

            client_model.fusion.load_state_dict(self.fusion.state_dict())

        client_model.setItemCommonality(self.item_commonality)

        client_model = client_model.to(self.device)
        client_optimizer = self._set_optimizer(client_model)

        return client_model, client_optimizer

    def _train_one_batch(self, batch, *args, **kwargs):
        txt_features = self.t_feat.to(self.device)
        vis_features = self.v_feat.to(self.device)

        batch_data = batch.to(self.device)
        user, poss, negs = batch_data[0], batch_data[1], batch_data[2]
        # construct ratings according to the interaction of positive and negative items
        items = torch.cat([poss, negs])
        ratings = torch.zeros(items.size(0), dtype=torch.float32)
        ratings[:poss.size(0)] = 1
        ratings = ratings.to(self.device)
        users = torch.full_like(ratings, user[0], dtype=torch.long).to(self.device)

        model, optimizer = args[0], args[1]

        optimizer.zero_grad()
        pred = model(users, items, txt_features, vis_features)
        loss = self.calculate_loss(pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model, optimizer, loss

    def _store_client_model(self, *args, **kwargs):
        user, client_model = args

        user_dict = copy.deepcopy(client_model.to('cpu').state_dict())
        self.client_models[user] = copy.deepcopy(user_dict)

        upload_params = {
            name: param.grad.clone() for name, param in client_model.fusion.named_parameters() if param.grad is not None
        }
        upload_params['item_commonality.weight'] = user_dict['item_commonality.weight'].clone()

        for key in user_dict.keys():
            if not any(sub in key for sub in ['router']):
                del self.client_models[user][key]
            else:
                if key in upload_params.keys():
                    del upload_params[key]

        return upload_params

    def _aggregate_params(self, *args, **kwargs):
        participant_params = args[0]

        num_participants = len(participant_params)

        self.fusion.train()
        self.optimizer.zero_grad()

        grad_accumulator = {}
        id_embed_weight_sum = None

        for user, param_dict in participant_params.items():

            w = self.weights[user] / self.model.n_items

            for name, param in param_dict.items():
                if name == 'item_commonality.weight':
                    id_embed_weight_sum = param if id_embed_weight_sum is None else id_embed_weight_sum + param
                else:
                    if name in grad_accumulator:
                        grad_accumulator[name] += param
                    else:
                        grad_accumulator[name] = param

        for name, param in self.fusion.named_parameters():
            if name in grad_accumulator:
                param.grad = grad_accumulator[name].to(param.device) / num_participants

        self.optimizer.step()

        self.item_commonality.weight.data = id_embed_weight_sum / num_participants

    def calculate_loss(self, interaction, *args, **kwargs):
        pred, truth = interaction, args[0]

        loss = self.crit(pred, truth)

        return loss
