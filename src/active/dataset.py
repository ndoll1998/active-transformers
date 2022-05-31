from torch.utils.data import Dataset, Subset
from itertools import filterfalse, chain
from random import sample
from typing import Sequence

class ActiveDataset(Subset):
    """ Dataset to simulate Active Learning setups by partitioning the source dataset into a labeled and unlabeled subset. 
        Adds functionality to activate/label specific datapoints from the given source dataset during the Active Learning loop.

        Args:
            dataset (Dataset): source or pool dataset containing all candidate data points
            labeled (Sequence[int]): 
                indices referencing data points in the pool that are initially labeled. 
                Defaults to an empty list, meaning the labeled dataset is empty.
    """

    def __init__(self, 
        dataset:Dataset,
        labeled:Sequence[int] =[]
    ) -> None:
        # initialize subset
        super(ActiveDataset, self).__init__(
            dataset=dataset,
            indices=labeled
        )

    @property
    def labeled_idx(self) -> Sequence[int]:
        """ List of indices referencing labeled data points in the source dataset. """
        return self.indices

    @property
    def unlabeled_idx(self) -> Sequence[int]:
        """ List of indices referencing unlabeled data points in the source dataset.
            The indices are sorted in ascending order.
        """
        # filter out all un-labeled indices
        idx = filterfalse(
            set(self.labeled_idx).__in__,
            range(len(self.dataset)
        )
        # and return it
        return list(idx)

    @property
    def labeled_data(self) -> Subset:
        """ Labeled subset of the source dataset. """
        return self

    @property
    def unlabeled_data(self) -> Subset:
        """ Un-labeld subset of the source dataset. """
        # build 
        return Subset(
            dataset=self.dataset,
            indices=self.unlabeled_idx
        )

    def label(self, indices:Sequence[int]) -> None:
        """ Activate/label specific data points of the unlabeled dataset. 
            
            Args:
                indices (Sequence[int]): 
                    indices referencing data points in the unlabeled dataset which are 
                    to be labeled.
        """
        
        # convert from unlabeld to source indices
        indices = map(self.unlabeled_idx.__getitem__, indices)
        # update labeled indices
        self.indices = list(chain(self.indices, indices))

    def label_random(self, n:int) -> None:
        """ Activate/label n random data points of the unlabeled dataset. 

            Args:
                n (int): the number of data points to label
        """
        # select random elements to label
        indices = sample(self.unlabeld_indices, n)
        self.label(indices=indices)
