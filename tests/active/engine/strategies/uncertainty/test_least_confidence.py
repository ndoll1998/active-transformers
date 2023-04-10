# no cuda
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from active.engine.strategies.uncertainty.least_confidence import LeastConfidence
# import utils
from active.utils.data import NamedTensorDataset
from tests.utils.modeling import PseudoModel
from tests.utils.strategy import RunStrategyTests, _test_strategy_behavior

class TestLeastConfidence(RunStrategyTests):

    def create_strategy(self, model):
        return LeastConfidence(model)

    def test_behavior(self):
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
                [
                    # somewhat certain
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
