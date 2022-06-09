# import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import ignite
from ignite.engine import Engine
from ignite.handlers.stores import EpochOutputStore
# others
from .utils import (
    map_tensors, 
    concat_tensors, 
    default_collate_drop_labels
)
from functools import partial
from abc import ABC, abstractmethod
from typing import Any, Sequence

class AbstractStrategy(Engine, ABC):
    """ Abstract base class for active learning strategies.
        Subclasses must define the following abstract methods:
            - process: executed by the ignite engine
            - sample: selects samples to label given the concatenated output of the engine run
    """

    def __init__(self):
        # initialize engine
        super(AbstractStrategy, self).__init__(type(self).process)
        # add output storage handler to self
        self.output_store = EpochOutputStore(
            # detach from gradient and move to cpu for storing
            output_transform=partial(map_tensors, fn=lambda t: t.detach().to('cpu'))
        )
        self.output_store.attach(self)

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
        # create dataloader
        loader = DataLoader(
            pool,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_collate_drop_labels
        )
        # reset output store and run the engine
        self.output_store.reset()
        self.run(loader)
        # get the recorded outputs
        output = self.output_store.data
        output = concat_tensors(output)
        # sample indices and check the number of indices
        indices = self.sample(output, query_size)
        assert len(indices) == min(query_size, len(pool))
        # return the sampled indices
        return indices
