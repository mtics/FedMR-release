import os

import numpy as np
import torch

from common.trainer import Trainer
from utils.utils import sampleClients


class FederatedTrainer(Trainer):

    def __init__(self, config, model, mg=False):
        super(FederatedTrainer, self).__init__(config, model, mg)

        self.global_model = None
        self.client_models = {}
        self.optimizers = {}

        self.last_participants = None

        self.weights = None

        if config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            # if file exist?
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(
                    torch.FloatTensor)
            if os.path.isfile(t_feat_file_path):
                self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(
                    torch.FloatTensor)

            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):

        # Check before training
        assert hasattr(self, 'model'), 'Please specify the model'
        if not self.req_training:
            return 0.0, []

        # Randomly select a subset of clients
        sampled_clients = sampleClients(
            list(train_data.user_set), self.config['clients_sample_strategy'],
            self.config['clients_sample_ratio'], self.last_participants
        )

        # Store the selected clients for the next round
        self.last_participants = sampled_clients
        self.weights = {user: 0 for user in range(self.model.n_users)}

        participant_params = {}
        total_loss, user_losses = 0, []
        for user in sampled_clients:
            client_loader = train_data.loaders[user]
            client_model, client_optimizer = self._set_client(user, epoch_idx)

            client_losses = []
            client_model.train()
            for epoch in range(self.config['local_epochs']):

                client_loss = 0
                for batch_idx, batch in enumerate(client_loader):
                    client_model, client_optimizer, loss = self._train_one_batch(batch, client_model, client_optimizer)

                    if self._check_nan(loss):
                        self.logger.info(
                            'NaN Loss exists at the [Batch:{} of {}-th Inner Epoch at {}-th user of {}-th outer loop]'.format(
                                batch_idx, epoch, user, epoch_idx))
                        return loss, torch.tensor(0.0)

                    client_loss += loss.item()

                client_losses.append(client_loss / len(client_loader))

                if epoch > 0 and abs(client_losses[-1] - client_losses[-2]) / (
                        client_losses[-1] + 1e-6) < self.config['tol']:
                    break

            total_loss += client_losses[-1]
            user_losses.append(client_losses[-1])

            participant_params[user] = self._store_client_model(user, client_model)

        # Aggregate the client model parameters in the server side
        self._aggregate_params(participant_params)

        # Update the model hyperparameters
        self._update_hyperparams(epoch_idx)

        return total_loss / len(sampled_clients), user_losses

    def _set_client(self, *args, **kwargs):
        pass

    def _train_one_batch(self, batch, *args, **kwargs):
        pass

    def _aggregate_params(self, *args, **kwargs):
        pass

    def _store_client_model(self, *args, **kwargs):
        pass

    def _update_hyperparams(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):

        t_feat, v_feat = None, None
        if self.config['is_multimodal_model']:
            t_feat = self.t_feat.to(self.device)
            v_feat = self.v_feat.to(self.device)

        metrics = None
        for user, loader in eval_data.loaders.items():
            client_model, _ = self._set_client(user, 1)
            client_model.eval()

            batch_scores = []
            for batch_idx, batch in enumerate(loader):
                batch = batch[1].to(self.device)
                scores = client_model.full_sort_predict(batch, t_feat, v_feat)
                mask = batch[1]
                scores[mask] = -float('inf')
                _, indices = torch.topk(scores, k=max(self.config['topk']))
                batch_scores.append(indices)

            client_metrics = self.evaluator.evaluate(batch_scores, loader, is_test=is_test, idx=idx)

            if metrics is None:
                metrics = client_metrics
            else:
                for key in client_metrics.keys():
                    metrics[key] += client_metrics[key]

        for key in metrics.keys():
            metrics[key] = metrics[key] / len(eval_data.loaders)

        return metrics
