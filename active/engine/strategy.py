# import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import ignite
from ignite.engine import Engine, Events
from ignite.handlers.stores import EpochOutputStore
# import utils
from active.utils.data import (
    map_tensors,
    concat_tensors,
    default_collate_drop_labels
)
from functools import partial
from abc import ABC, abstractmethod
from typing import Any, Sequence, Optional

class ActiveStrategy(Engine, ABC):
    """ Abstract base class for active learning strategies.
        Subclasses must define the following abstract methods:
            - `process`: executed by the ignite engine
            - `sample`: selects samples to label given the concatenated output of the engine run
    """

    def __init__(self):
        # initialize engine
        super(ActiveStrategy, self).__init__(type(self).process)
        # add output storage handler to self
        self.output_store = EpochOutputStore(
            # detach from gradient and move to cpu for storing
            output_transform=partial(map_tensors, fn=lambda t: t.detach().cpu())
        )
        self.output_store.attach(self)
        # last selected indices
        self._indices:Sequence[int] = None

    @property
    def output(self) -> Any:
        """ Output of the strategy engine. Returns output corresponding to the
            (unlabeled) data points of the pool last processed by the strategy.
            Depending on the type of strategy this corresponds to a tpye of
            representation (i.e. uncertainty scores, gradient embedding, etc.).
            Returns `None` before the first exection of the strategy.
        """
        # empty data store, i.e. strategy not executed yet
        if len(self.output_store.data) == 0:
            return None
        # get the recorded outputs and concatenate them
        return concat_tensors(self.output_store.data)

    @property
    def selected_indices(self) -> Sequence[int]:
        """ List of indices last selected by the strategy. Indices reference items
            in the unlabeled data pool last processed by the strategy. Returns
            `None` before the first execution.
        """
        return self._indices

    @abstractmethod
    def process(self, batch:Any) -> Any:
        """ Process a given batch. Outputs of this function will
            be concatenated and passed to the `sample` function.

            Args:
                batch (Any): a batch of input samples from the unlabeled dataset

            Returns:
                output (Any): output to the batch
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self, output:Any, query_size:int) -> Sequence[int]:
        """ Sample examples given the concatenated output of the process function.

            Args:
                output (Any): the concatenated output of the engine run, i.e. the process function.
                query_size (int): the number of samples to select for labeling

            Returns:
                indices (Sequence[int]): indices of samples in the unlabeled pool which to label next
        """
        raise NotImplementedError()

    def _sample(self, query_size:int, pool:Dataset) -> None:
        """ Event Handler called on `EPOCH_COMPLETED` (Note that there is no reason
            to run a strategy for more than one epoch). Calls the `sample` function
            and stores the output in the `_indices` attribute of the strategy.

            This way tracked time stored in `state.times[`COMPLETED`]` includes
            processing and sampling.

            Only called when engine is run by calling `query` function. Not called
            when engine is directly executed by `run` function.

            Args:
                query_size (int): number of data points to sample from pool
                pool (Dataset): pool of data points from which to sample
        """
        # call sample function and check indices
        indices = list(self.sample(self.output, query_size))
        assert len(indices) == min(query_size, len(pool))
        # set attribute
        self._indices = indices

    def dataloader(self, data:Dataset, **kwargs) -> DataLoader:
        """ Create the dataloader for a given dataset with some specific configuration.

            Args:
                data (Dataset): dataset to use
                **kwargs (Any): keyword arguments passed to the dataloader

            Returns:
                loader (DataLoader): dataloader from given dataset and configuration
        """
        return DataLoader(data, **kwargs)

    def query(
        self,
        pool:Dataset,
        query_size:int,
        batch_size:int
    ) -> Sequence[int]:
        """ Query samples to label using the strategy

            Args:
                pool (Dataset): pool of data points from which to sample
                query_size (int): number of data points to sample from pool
                batch_size (int): batch size used to process the pool dataset

            Returns:
                indices (Sequence[int]): indices of selected data points in pool
        """
        # check query size
        assert query_size <= len(pool), "Query size (%i) larger than pool (%i)" % (query_size, len(pool))
        # create dataloader
        loader = self.dataloader(
            pool,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_collate_drop_labels
        )
        # add sample event handler which updates the `_indices` attribute
        handler = partial(self._sample, query_size=query_size, pool=pool)
        with self.add_event_handler(Events.EPOCH_COMPLETED, handler):
            # reset output store and run the engine
            self.output_store.reset()
            self.run(loader)
        # return the sampled indices
        return self._indices
