import torch
# import uncertainty strategies
from src.active.strategies.uncertainty import (
    LeastConfidence,
    PredictionEntropy
)
# import utils
from tests.common import PseudoModel, NamedTensorDataset
from tests.active.strategies.common import _test_strategy_behavior

class TestUncertaintyStrategies:
    """ Test cases for Uncertainty Strategies """
 
    def test_least_confidence(self):
        # create least confidence strategy
        strategy = LeastConfidence(model=PseudoModel())
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
        # test strategy
        _test_strategy_behavior(
            strategy=strategy,
            pool=pool,
            expected_order=[0, 1, 2]
        )
