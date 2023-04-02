import torch
from torch.utils.data import DataLoader
import transformers
from math import ceil

class RepeatDataLoaderWrapper(DataLoader):

    def __init__(self, loader:DataLoader, repeat:int):
        self.loader = loader
        self.repeat = repeat

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    def __iter__(self):
        for i in range(self.repeat):
            yield from self.loader

    def __len__(self):
        return len(self.loader) * self.repeat

class CustomTrainer(transformers.Trainer):

    def get_train_dataloader(self) -> DataLoader:
        # get train dataloader
        train_loader = super().get_train_dataloader()

        # repeat data loader if not enough samples
        if hasattr(self.args, 'min_epoch_length') and (len(train_loader) < self.args.min_epoch_length):
            n = ceil(self.args.min_epoch_length / len(train_loader))
            return RepeatDataLoaderWrapper(train_loader, repeat=n)

        # otherwise use the original
        return train_loader

