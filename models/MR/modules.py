import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_normal_initialization
from models.MR.experts import SumExpert, MLPExpert, MultiHeadAttentionExpert, GateExpert


class GatingNetwork(torch.nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0., latent_dim=128):
        super(GatingNetwork, self).__init__()

        self.fc1 = nn.Linear(in_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        weights = F.softmax(out, dim=1)

        # 检查weights的维度，如果大于一维，则计算其平均值
        if len(weights.size()) > 1:
            weights = torch.mean(weights, dim=0)

        return weights.unsqueeze(0)


class TopKModule(torch.autograd.Function):

    @staticmethod
    def forward(ctx, scores, topk):
        top_n_scores, top_n_indices = torch.topk(scores, topk, dim=1)
        ctx.save_for_backward(scores, top_n_scores, top_n_indices)
        ctx.topk = topk
        return top_n_scores, top_n_indices

    @staticmethod
    def backward(ctx, grad_top_n_scores, grad_top_n_indices):
        scores, top_n_scores, top_n_indices = ctx.saved_tensors
        grad_scores = torch.zeros_like(scores)
        grad_scores.scatter_(1, top_n_indices, grad_top_n_scores)
        return grad_scores, None


class SwitchingFusionModule(nn.Module):

    def __init__(self, in_dim, embed_dim, dropout=0., latent_dim=128):
        super(SwitchingFusionModule, self).__init__()

        self.idx = -1

        self.router = GatingNetwork(in_dim * 3, 3, dropout, latent_dim)

        self.experts = nn.ModuleList([
            SumExpert(),
            MLPExpert(embed_dim),
            GateExpert(embed_dim, embed_dim)
            # MultiHeadAttentionExpert(embed_dim, 8)
        ])

    def forward(self, x, y, z):
        # 将x,y,z分别送入三个专家模块，然后将三个模块的输出拼接在一起，送入self.router进行路由选择
        outs = [expert(x, y, z) for expert in self.experts]

        c = torch.cat(outs, dim=1)

        scores = self.router(c)

        weights = torch.softmax(scores, dim=1)

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out += weights[0, i] * outs[i]

        return out


class FusionLayer(nn.Module):

    def __init__(self, in_dim, fusion_module='moe', latent_dim=128):
        super(FusionLayer, self).__init__()

        self.id_affine = nn.Linear(in_dim, latent_dim)
        self.txt_affine = nn.Linear(in_dim, latent_dim)
        self.vis_affine = nn.Linear(in_dim, latent_dim)

        if fusion_module == 'moe':
            self.fusion = SwitchingFusionModule(latent_dim, latent_dim, dropout=0.5, latent_dim=latent_dim)
        elif fusion_module == 'sum':
            self.fusion = SumExpert()
        elif fusion_module == 'mlp':
            self.fusion = MLPExpert(latent_dim)
        elif fusion_module == 'attention':
            self.fusion = MultiHeadAttentionExpert(latent_dim, 8)
        elif fusion_module == 'gate':
            self.fusion = GateExpert(latent_dim, latent_dim)
        else:
            raise ValueError('Invalid fusion module, currently only support: moe, sum, mlp, and attention.')

    def forward(self, id_feat, txt_feat, vis_feat):

        id_feat = self.id_affine(id_feat)
        txt_feat = self.txt_affine(txt_feat)
        vis_feat = self.vis_affine(vis_feat)

        return self.fusion(id_feat, txt_feat, vis_feat)


class MR(GeneralRecommender):

    def __init__(self, config, dataloader):
        super(MR, self).__init__(config, dataloader)
        self.config = config

        self.item_pool = torch.tensor(range(self.n_items))
        self.id_embed = nn.Embedding(self.n_items, config['embedding_size'])

        self.fusion = FusionLayer(config['embedding_size'], fusion_module=config['fusion_module'],
                                  latent_dim=config['latent_size'])

        self.predictor = nn.Linear(config['embedding_size'], 1)
        self.logistic = nn.Sigmoid()

        self.apply(xavier_normal_initialization)

    def set_id_embed(self, id_embed):
        self.id_embed.weight.data = copy.deepcopy(id_embed.weight.data)

    def forward(self, item_indices, txt_embed, vision_embed):
        # id_embed = self.id_embed(self.item_pool)
        id_embed = self.id_embed(item_indices)
        txt = txt_embed[item_indices]
        vision = vision_embed[item_indices]

        out = self.fusion(id_embed, txt, vision)
        out = self.predictor(out)
        out = self.logistic(out)

        return out

    def full_sort_predict(self, interaction, *args, **kwargs):
        txt_embed, vis_embed = args[0], args[1]

        # Check whether self.item_pool is on the same device as ratings
        items = torch.arange(self.n_items).to(self.device)

        scores = self.forward(items, txt_embed, vis_embed)

        return scores.view(-1)
