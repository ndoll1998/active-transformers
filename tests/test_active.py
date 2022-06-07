import torch
from torch.utils.data import Subset
# import active learning components
from src.active.loop import ActiveLoop
from src.active.heuristics.heuristic import ActiveHeuristic
from src.active.heuristics.random import Random
from src.active.heuristics.uncertainty import LeastConfidence
# helpers
from .utils import NamedTensorDataset
from types import SimpleNamespace
from typing import List

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
            heuristic=Random()
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

class TestHeuristics:
    """ Test cases for Heuristics """

    def pseudo_model(self, **kwargs):
        """ Pseudo Model used for testing Heustics. Returns input arguments as 
            namespace to allow access using dot-operator 
        """
        return SimpleNamespace(**kwargs)

    def _test_heuristic_behavior(self, 
        heuristic:ActiveHeuristic,
        pool:NamedTensorDataset,
        expected_order:List[int]
    ):
        """ Helper function to test the behaviour of a heuristic. 

            Args:
                heuristic (ActiveHeuristic): heuristic to test
                pool (NamedTensorDataset): pool of data points
                expected_order (List[int]): expected selection order of data points in the pool

            Throws AssertionError if heuristic doesn't match expectation
        """
        # create active loop
        loop = ActiveLoop(
            pool=pool,
            batch_size=len(pool),
            query_size=1,
            heuristic=heuristic,
            init_heuristic=heuristic
        )
        # make sure sampling matches expectation
        for expected_idx, data in zip(expected_order, loop):
            sampled_idx = data.indices[0]
            assert expected_idx == sampled_idx
            
    def test_least_confidence(self):
        # create a heuristic
        heuristic = LeastConfidence(model=self.pseudo_model)
        # create a sample pool
        pool = NamedTensorDataset(
            logits=torch.FloatTensor([
                [
                    # very uncertain
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]
                ],
                # somewhat certain
                [
                    [1.0, 0.0, 0.8],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    # very certain
                    [-1.0, 1.0, -1.0],
                    [1.0, -1.0, -1.0],
                    [-1.0, -1.0, 1.0]
                ],
            ]),
            attention_mask=torch.BoolTensor([
                [True, True, False],
                [True, False, False],
                [True, True, True]
            ])
        )
        # test heuristic
        self._test_heuristic_behavior(
            heuristic=heuristic,
            pool=pool,
            expected_order=[0, 1, 2]
        )
