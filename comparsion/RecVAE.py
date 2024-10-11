import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from utils.trainer import CentralTrainer
from utils.vae import Encoder, Decoder
from utils import evaluation


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3 / 20, 3 / 4, 1 / 10]):
        super(CompositePrior, self).__init__()

        self.mixture_weights = mixture_weights

        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)

        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)

        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        # post_mu, post_logvar = self.encoder_old(x, 0)
        post_mu, post_logvar = self.encoder_old(x)

        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]

        density_per_gaussian = torch.stack(gaussians, dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1, drop_rate=0.5):
        super(Encoder, self).__init__()

        self.dropout_rate = drop_rate

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]

        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


class RecVAE(nn.Module):
    def __init__(self, args):
        super(RecVAE, self).__init__()

        self.args = args
        self.beta = args.beta

        input_dim = args.num_items
        latent_dim = args.latent_dim
        hidden_dim = int((input_dim + latent_dim) / 2)

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim, drop_rate=args.dropout)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings):
        mu, logvar = self.encoder(user_ratings)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)

        return x_pred, mu, logvar, z

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))

    def calc_loss(self, pred, truth, mu, logvar, z):
        mll = (F.log_softmax(pred, dim=-1) * truth).sum(1).mean()
        kld = (log_norm_pdf(z, mu, logvar) - self.prior(truth, z)).sum(dim=-1).mul(self.beta).mean()
        negative_elbo = -(mll - kld)

        return negative_elbo


class RecVAETrainer(CentralTrainer):

    def __init__(self, args):
        self.model = RecVAE(args)
        super(RecVAETrainer, self).__init__(args)

        print(self.model)

        self.beta = args.beta
        self.optimizer = self.set_optimizer()

        # Move model to the specified device
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    def set_optimizer(self, *args, **kwargs):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_reg)

    def _train_one_batch(self, batch, *args, **kwargs):
        batch = self.fabric.to_device(batch)

        self.optimizer.zero_grad()
        pred, mu, logvar, z = self.model(batch)
        loss = self.calculate_loss(pred, batch, mu, logvar, z)

        self.fabric.backward(loss)

        self.optimizer.step()

        return loss.item()

    def calculate_loss(self, *args, **kwargs):
        pred, truth, mu, logvar, z = args[0], args[1], args[2], args[3], args[4]

        loss = self.model.calc_loss(pred, truth, mu, logvar, z)

        return loss

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
