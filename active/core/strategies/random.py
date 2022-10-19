import torch
from torch.utils.data import Dataset
from .strategy import AbstractStrategy
from typing import Sequence

class Random(AbstractStrategy):

    def process(self, batch):
        pass

    def sample(self, output, num_samples):
        pass

    def query(
        self,
        pool:Dataset,
        query_size:int,
        batch_size:int
    ) -> Sequence[int]:
        # sample random indices within the pool
        perm = torch.randperm(len(pool))
        return perm[:query_size]
