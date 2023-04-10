# import torch and data utils
import torch
from torch.utils.data import Dataset
# import base class
from .common.strategy import ActiveStrategy
# other utils
from typing import Sequence

class Random(ActiveStrategy):

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
