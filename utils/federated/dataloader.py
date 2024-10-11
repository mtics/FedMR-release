import pandas

import utils.dataloader as loader_utils


class FederatedDataLoader(object):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data which is loaded by
    :class:`~recbole.data.interaction.Interaction` when it is iterated.
    And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        dataset (Dataset): The dataset of this dataloader.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
        real_time (bool): If ``True``, dataloader will do data pre-processing,
            such as neg-sampling and data-augmentation.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        batch_size (int): The max interaction number for all batch.
    """

    def __init__(self, config, dataset, batch_size=1, neg_sampling=False, shuffle=False, stage='train',
                 additional_dataset=None):
        self.config = config
        self.dataset = dataset

        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.neg_sampling = neg_sampling

        self.additional_dataset = additional_dataset

        self.stage = stage

    def _get_federated_loader(self):
        data_loader = None
        if self.stage == 'train':
            data_loader = self._get_train_loader()
        elif self.stage == 'eval' or self.stage == 'test':
            data_loader = self._get_eval_loader()

        return data_loader

    def _get_train_loader(self):
        user_datasets, user_loader = {}, {}
        for user_id in self.dataset[self.dataset.uid_field].unique():
            user_datasets[user_id] = self.dataset[self.dataset[self.dataset.uid_field] == user_id]
            user_loader[user_id] = loader_utils.TrainDataLoader(self.config, user_datasets[user_id],
                                                                batch_size=self.config['train_batch_size'],
                                                                shuffle=self.shuffle)
        return user_loader

    def _get_eval_loader(self):
        assert self.additional_dataset is not None, 'additional_dataset should not be None in eval dataloader'
        assert isinstance(self.additional_dataset, pandas.DataFrame), 'additional_dataset should be a DataFrame'

        user_datasets, user_loader = {}, {}
        for user_id in self.dataset[self.dataset.uid_field].unique():
            user_additional_data = self.additional_dataset[
                self.additional_dataset[self.additional_dataset.uid_field] == user_id]
            user_datasets[user_id] = self.dataset[self.dataset[self.dataset.uid_field] == user_id]
            user_loader[user_id] = loader_utils.EvalDataLoader(self.config, user_datasets[user_id],
                                                               batch_size=self.config['eval_batch_size'],
                                                               additional_dataset=user_additional_data)
        return user_loader
