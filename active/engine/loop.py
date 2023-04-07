# import torch and data utils
import torch
from torch.utils.data import Dataset, Subset
# import abstract strategy
from .strategy import ActiveStrategy
# import other utils
from itertools import repeat, filterfalse
from typing import Sequence, Union
from math import ceil

class ActiveLoop(object):
    """ Active Learning Loop running the basic loop selecting
        samples from the data pool by applying a strategy.

        Can be used as an iterator, running until either the data
        pool or the sequence of strategies to apply is exhausted.

        Args:
            pool (Dataset): the pool dataset
            batch_size (int): batch size with which to process the data pool
            query_size (int): number of samples to select
            strategy (Union[ActiveStrategy, Sequence[ActiveStrategy]]):
                the Active Learning Strategy to use for selection. Also supports
                a sequence of strategies in which case the strategy at index i is
                applied in the i-th Active Learning iteration.
    """

    def __init__(
        self,
        pool:Dataset,
        batch_size:int,
        query_size:int,
        strategy:Union[ActiveStrategy, Sequence[ActiveStrategy]],
    ) -> None:
        self.pool = Subset(
            dataset=pool,
            indices=list(range(len(pool)))
        )
        self.batch_size = batch_size
        self.query_size = query_size
        # sequence of strategies
        self.strategies = repeat(strategy) if isinstance(strategy, ActiveStrategy) else strategy
        self.strategies_iter = None # set in reset

        # cache
        self._cached_strategy = None # updated in step

    @property
    def strategy(self) -> Union[None, ActiveStrategy]:
        """ Strategy applied in the last iteration of the loop """
        return self._cached_strategy

    def reset(self) -> None:
        # reset data pool
        self.pool = Subset(
            dataset=self.pool.dataset,
            indices=list(range(len(self.pool.dataset)))
        )
        # reset strategy iterator and cached strategy
        self.strategies_iter = iter(self.strategies)
        self._cached_strategy = None
        # return loop instance
        return self

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

    def step(self) -> Subset:
        # check if the data pool is exhausted
        if len(self.pool) == 0:
            raise StopIteration
        # get the strategy to apply in the next iteration
        # already raises stop-iteration if exhausted
        self._cached_strategy = next(self.strategies_iter)
        # select samples using strategy
        indices = self._cached_strategy.query(
            pool=self.pool,
            query_size=min(self.query_size, len(self.pool)),
            batch_size=self.batch_size
        )
        # update current iteration
        return self.update([self.pool.indices[i] for i in indices])

    def __len__(self) -> int:
        return ceil(len(self.pool) / self.query_size)

    def __iter__(self):
        return self.reset()

    def __next__(self):
        return self.step()
