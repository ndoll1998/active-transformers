import torch
from torch.utils.data import Subset
# iport active learning components
from src.active.loop import ActiveLoop
from src.active.strategies.random import Random
# import helpers
from tests.common import NamedTensorDataset

class TestActiveLoop:
    """ Test cases for the `ActiveLoop` class """    

    def test_unique_sampling(self):
        """ Test if the samples selected by the active loop are unique, i.e.
            each data point in the pool is chosen at most once.
        """

        # create sample dataset
        pool = NamedTensorDataset(input_ids=torch.arange(100))

        # create active loop
        loop = ActiveLoop(
            pool=pool,
            batch_size=4,
            query_size=4,
            strategy=Random()
        )

        # get all samples datasets
        sampled_datasets = list(loop)

        # check the number of sampled datasets
        assert len(sampled_datasets) == len(pool) // loop.query_size
        # make sure all sampled datasets are subsets
        # this is needed for further checks
        assert all(isinstance(data, Subset) for data in sampled_datasets)
        # make sure all sampled sets are subsets of the pool dataset
        assert all(data.dataset == pool for data in sampled_datasets)

        for data in sampled_datasets:
            elements = [data[i]['input_ids'].item() for i in range(len(data))]
            indices = data.indices
            # by construction of the pool dataset the data points
            # are the indices within the pool dataset
            assert set(elements) == set(indices)
            # also make sure all indices are unique
            assert len(set(indices)) == len(indices)

        # check for pairwise overlaps in the sampled subsets
        idx_sets = [set(data.indices) for data in sampled_datasets]
        for i, A in enumerate(idx_sets):
            for B in idx_sets[i+1:]:
                assert len(A & B) == 0

