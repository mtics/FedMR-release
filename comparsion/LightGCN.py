import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.trainer import CentralTrainer
from utils import evaluation


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding ** self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss


class LightGCN(nn.Module):

    def __init__(self, config):
        super(LightGCN, self).__init__()

        self.config = config
        self.adj_mat = config.fabric.to_device(config.adj_mat)

        self.num_items = config.n_items
        self.num_users = config.n_users

        self.num_layers = config.num_layers
        self.latent_dim = config.latent_size

        self.user_embedding = nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.latent_dim)

        self.logistic = nn.Sigmoid()

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embedding_list = [all_embeddings]
        for idx in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.adj_mat, all_embeddings)
            embedding_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embedding_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings,
                                                               [self.num_users, self.num_items], dim=0)

        return user_all_embeddings, item_all_embeddings


class LightGCNTrainer(CentralTrainer):

    def __init__(self, args):
        self.model = LightGCN(args)
        super(LightGCNTrainer, self).__init__(args)

        print(self.model)

        self.lr = self.args.lr
        self.beta = args.beta
        self.optimizer = self.set_optimizer()

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Move model to the specified device
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    def set_optimizer(self, *args, **kwargs):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.l2_reg)

    def _train_one_batch(self, batch, *args, **kwargs):
        batch = self.fabric.to_device(batch)

        self.optimizer.zero_grad()
        user_all_embeddings, item_all_embeddings = self.model()
        loss = self.calculate_loss(user_all_embeddings, item_all_embeddings, batch)

        self.fabric.backward(loss)

        self.optimizer.step()

        return loss.item()

    def calculate_loss(self, *args, **kwargs):

        user_all_embeddings, item_all_embeddings, batch = args[0], args[1], args[2]
        users, items, ratings = batch
        pos_idx = items[ratings == 1]
        neg_idx = items[ratings == 0]

        u_embeddings = user_all_embeddings[users]
        i_embeddings = item_all_embeddings[items]
        pos_embeddings = item_all_embeddings[pos_idx]
        neg_embeddings = item_all_embeddings[neg_idx]

        all_socres = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        # calculate BPR Loss
        # pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        # neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        pos_scores = all_socres[pos_idx]
        neg_scores = all_socres[neg_idx]
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.model.user_embedding(users)
        pos_ego_embeddings = self.model.id_embed(pos_idx)
        neg_ego_embeddings = self.model.id_embed(neg_idx)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=False,
        )

        loss = mf_loss + self.beta * reg_loss

        return loss

    def predict(self, users, items):
        user_all_embeddings, item_all_embeddings = self.model()
        u_embeddings, item_embeddings = user_all_embeddings[users], item_all_embeddings[items]
        scores = torch.mul(u_embeddings, item_embeddings).sum(dim=1)
        return scores

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
            users, items, ratings = batch

            scores = self.predict(users, items)

            if metrics == None:
                metrics = evaluation.recalls_and_ndcgs_for_ks(scores, ratings)
            else:
                values = evaluation.recalls_and_ndcgs_for_ks(scores, ratings)
                for key in metrics.keys():
                    metrics[key] += values[key]

        for key in metrics.keys():
            metrics[key] = metrics[key] / len(eval_data)

        return metrics['Recall@{}'.format(self.args.top_k)], metrics['NDCG@{}'.format(self.args.top_k)]
