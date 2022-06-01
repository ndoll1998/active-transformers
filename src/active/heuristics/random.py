import torch
from torch.utils.data import Dataset
from .heuristic import ActiveHeuristic

class Random(ActiveHeuristic):

    def __init__(self):
        # random is model indipendent
        super(Random, self).__init__(model=None)

    def step(self, engine, batch) -> torch.Tensor:
        # get batch size and return random scores
        batch_size = batch['input_ids'].size(0)
        return torch.rand(batch_size)
