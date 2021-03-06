import torch
from torch.utils.data import Dataset, Subset
from .strategies.strategy import AbstractStrategy
from .strategies.random import Random
from itertools import filterfalse
from typing import Sequence

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

    def apply_strategy(self, strategy:AbstractStrategy) -> Subset:
        """ """
        # select samples using strategy
        indices = strategy.query(
            pool=self.pool,
            query_size=self.query_size,
            batch_size=self.batch_size
        )
        indices = [self.pool.indices[i] for i in indices]
        # get the data points and remove them from the pool
        data = Subset(
            dataset=self.pool.dataset,
            indices=indices
        )
        self.pool.indices = list(filterfalse(
            set(indices).__contains__,
            self.pool.indices
        ))
        # return data
        return data

    def init_step(self) -> Subset:
        return self.apply_strategy(self.init_strategy)
    
    def step(self) -> Subset:
        return self.apply_strategy(self.strategy)

    def __iter__(self):
        # apply initial heuristic
        yield self.init_step()
        # apply heuristic unitl pool is empty
        while len(self.pool) >= self.query_size:
            yield self.step()
