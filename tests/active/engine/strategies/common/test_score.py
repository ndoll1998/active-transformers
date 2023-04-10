import pytest
import torch
from torch.utils.data import TensorDataset
from active.engine.strategies.common.score import ScoreBasedStrategy
from tests.utils.strategy import _test_strategy_behavior

class TestScoreBasedStrategy():

    def test_score_based_strategy_argmax(self):
        """ Test deterministic Score-based strategy """

        class Strategy(ScoreBasedStrategy):
            process = lambda self, batch: batch[0]

        _test_strategy_behavior(
            strategy=Strategy(),
            pool=TensorDataset(torch.FloatTensor([0, 0, 0.5, 0.6, 0, 0.9])),
            expected_order=[5, 3, 2],
            batch_size=2
        )


    @pytest.mark.parametrize('exec_number', range(10))
    def test_score_based_strategy_sampling(self, exec_number):
        """ Test score-based strategy with random sampling"""

        class Strategy(ScoreBasedStrategy):
            process = lambda self, batch: batch[0]

        # create strategy and pool
        strategy = Strategy(random_sample=True)
        pool = torch.FloatTensor([0, 0, 0.5, 0.6, 0, 0.9])

        # query
        idx = strategy.query(
            TensorDataset(pool),
            query_size=2,
            batch_size=2
        )
        # check expectation
        assert all(i.item() in {2, 3, 5} for i in idx)
