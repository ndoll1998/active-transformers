import torch
from torch.utils.data import Dataset, Subset
from .strategies.strategy import AbstractStrategy
from .strategies.random import Random
from itertools import filterfalse
from typing import Sequence
from math import ceil

class ActiveLoop(object):

    def __init__(
        self,
        pool:Dataset,
        batch_size:int,
        query_size:int,
        strategy:AbstractStrategy,
        init_strategy:AbstractStrategy =Random()
    ) -> None:
        self.pool = Subset(
            dataset=pool,
            indices=list(range(len(pool)))
        )
        self.batch_size = batch_size
        self.query_size = query_size
        self.strategy = strategy
        self.init_strategy = init_strategy

    def update(self, indices:Sequence[int]) -> Subset:
        # get the data points and remove them from the pool
        data = Subset(
            dataset=self.pool.dataset,
            indices=list(indices)
        )
        self.pool.indices = list(filterfalse(
            set(indices).__contains__,
            self.pool.indices
        ))
        # return data
        return data

    def apply_strategy(self, strategy:AbstractStrategy) -> Subset:
        """ """
        # select samples using strategy
        indices = strategy.query(
            pool=self.pool,
            query_size=min(self.query_size, len(self.pool)),
            batch_size=self.batch_size
        )
        return self.update([self.pool.indices[i] for i in indices])

    def init_step(self) -> Subset:
        return self.apply_strategy(self.init_strategy)
    
    def step(self) -> Subset:
        return self.apply_strategy(self.strategy)

    def __len__(self) -> int:
        return ceil(len(self.pool) / self.query_size)

    def __iter__(self):
        # apply initial heuristic
        yield self.init_step()
        # apply heuristic unitl pool is empty
        while len(self.pool) > 0:
            yield self.step()
