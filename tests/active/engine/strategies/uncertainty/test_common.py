# no cuda
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
# import uncertainty strategy
from active.engine.strategies.uncertainty.common import UncertaintyStrategy
# import utils
from active.utils.data import NamedTensorDataset
from tests.utils.modeling import PseudoModel

class TestUncertaintyStrategy:

    def test_label_ignoring(self):

        class TestStrategy(UncertaintyStrategy):
            def uncertainty_measure(self, probs):
                return probs.sum(dim=-1)

        # create least confidence strategy
        strategy = TestStrategy(
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
