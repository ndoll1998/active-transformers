import torch
from torch.utils.data import Subset
# iport active learning components
from active.engine.loop import ActiveLoop
from active.engine.strategies.random import Random
from active.utils.data import NamedTensorDataset
# import helpers
import pytest

class TestActiveLoop:
    """ Test cases for the `ActiveLoop` class """

    @pytest.mark.parametrize('exec_number', range(5))
    def test_unique_sampling(self, exec_number):
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
            strategy=Random(),
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

    def test_reset(self):

        # create sample dataset
        pool = NamedTensorDataset(input_ids=torch.arange(100))
        # create active loop
        loop = ActiveLoop(
            pool=pool,
            batch_size=4,
            query_size=4,
            strategy=Random()
        )

        # run loop and reset
        list(loop)
        loop.reset()
        # check
        assert len(loop.pool) == len(pool), "Expected pool to be reset"
        assert loop.strategy is None, "Expected strategy to be reset"

    def test_strategy_sequence(self):

        # strategies to apply in each active learning iteration
        strategies = [Random() for _ in range(5)]
        # create sample dataset
        pool = NamedTensorDataset(input_ids=torch.arange(100))
        # create active loop
        loop = ActiveLoop(
            pool=pool,
            batch_size=4,
            query_size=4,
            strategy=strategies,
        )

        # check if strategies match
        for s, _ in zip(strategies, loop):
            assert loop.strategy is s, "Mismatch of strategies"

    def test_exhaust_strategy_sequence(self):

        # strategies to apply in each active learning iteration
        strategies = [Random() for _ in range(5)]
        # create sample dataset
        pool = NamedTensorDataset(input_ids=torch.arange(100))
        # create active loop
        loop = ActiveLoop(
            pool=pool,
            batch_size=4,
            query_size=4,
            strategy=strategies,
        )

        # run loop manually
        loop.reset()
        for _ in strategies:
            loop.step()

        # test exit iteration
        try:
            loop.step()
            pytest.fail("Expected StopIteration by exhaustion of strategy sequence!")
        except StopIteration:
            pass

