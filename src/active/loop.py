import torch
from torch.utils.data import Dataset, Subset
from .heuristics.heuristic import ActiveHeuristic
from .heuristics.random import Random
from itertools import filterfalse
from typing import Sequence

class ActiveLoop(object):

    def __init__(
        self,
        pool:Dataset,
        batch_size:int,
        query_size:int,
        heuristic:ActiveHeuristic,
        init_heuristic:ActiveHeuristic =Random()
    ) -> None:
        self.pool = Subset(
            dataset=pool,
            indices=list(range(len(pool)))
        )
        self.batch_size = batch_size
        self.query_size = query_size
        self.heuristic = heuristic
        self.init_heuristic = init_heuristic

    def step_heuristic(self, heuristic:ActiveHeuristic) -> Subset:
        """ """
        # compute ranking scores
        scores = heuristic.compute_scores(
            dataset=self.pool,
            batch_size=self.batch_size
        )
        # get top-k indices
        indices = scores.topk(k=self.query_size).indices
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
        return self.step_heuristic(self.init_heuristic)
    
    def step(self) -> Subset:
        return self.step_heuristic(self.heuristic)

    def __iter__(self):
        # apply initial heuristic
        yield self.init_step()
        # apply heuristic unitl pool is empty
        while len(self.pool) >= self.query_size:
            yield self.step()
