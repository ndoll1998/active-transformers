import torch
# import uncertainty strategies
from src.active.strategies.uncertainty import (
    LeastConfidence,
    PredictionEntropy
)
# import utils
from src.active.utils.tensor import NamedTensorDataset
from tests.common import PseudoModel
from tests.active.strategies.common import _test_strategy_behavior

class TestUncertaintyStrategies:
    """ Test cases for Uncertainty Strategies """
 
    def test_label_ignoring(self):
        # create least confidence strategy
        strategy = LeastConfidence(
            model=PseudoModel(),
            ignore_labels=[0]
        )
        # create a sample pool
        pool = NamedTensorDataset(
            logits=torch.FloatTensor([
                [
                    # all predicted to be label 0
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]
                ],
                [
                    # none predicted to be label 0
                    [0.3, 0.0, 0.7],
                    [0.2, 0.8, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]),
            attention_mask=torch.BoolTensor([
                [True, True, True],
                [True, True, True],
            ])
        )
        # query strategy
        strategy.query(
            pool=pool,
            query_size=1,
            batch_size=2
        )
        # get uncertainty scores
        scores = strategy.output
        assert isinstance(scores, torch.FloatTensor), "Expected scores to be of type FloatTensor but got type %s" % str(type(scores))
        assert (scores.ndim == 1) and (scores.size(0) == 2), "Expected scores to have a single dimension of size 2 but got shape %s" % str(tuple(scores.size()))
        # check scores
        assert scores[0] == 0.0, "Should be zero because of label ignoring!"
        assert scores[1] > 0.0, "Shouldn't be zero"

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
