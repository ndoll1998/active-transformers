import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
# import active learning components
from src.active.loop import ActiveLoop
from src.active.strategies.strategy import AbstractStrategy
from src.active.strategies.uncertainty import (
    LeastConfidence,
    PredictionEntropy
)
from src.active.strategies.badge import (
    Badge,
    BadgeForSequenceClassification,
    BadgeForTokenClassification
)
from src.active.strategies.egl import (
    GoodfellowGradientNorm,
    EglForSequenceClassification
)
# import helpers
import pytest
import warnings
from tests.common import (
    NamedTensorDataset, 
    PseudoModel
)
from typing import List

class TestUncertaintyStratgies:
    """ Test cases for Uncertainty Strategies """

    def _test_strategy_behavior(self, 
        strategy:AbstractStrategy,
        pool:NamedTensorDataset,
        expected_order:List[int]
    ):
        """ Helper function to test the behaviour of a strategy. 

            Args:
                strategy (AbstractStrategy): strategy to test
                pool (NamedTensorDataset): pool of data points
                expected_order (List[int]): expected selection order of data points in the pool

            Throws AssertionError if strategy doesn't match expectation
        """
        # create active loop
        loop = ActiveLoop(
            pool=pool,
            batch_size=len(pool),
            query_size=1,
            strategy=strategy,
            init_strategy=strategy
        )
        # make sure sampling matches expectation
        for expected_idx, data in zip(expected_order, loop):
            sampled_idx = data.indices[0]
            assert expected_idx == sampled_idx
            
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
        self._test_strategy_behavior(
            strategy=strategy,
            pool=pool,
            expected_order=[0, 1, 2]
        )


class TestBadge:
    """ Tests for Badge Strategy """

    def test_initialization_warnings(self):
        """ Test Badge initialization warnings """
        # create bert config
        config = transformers.BertConfig()
        # warning message to test for
        message = "Pipeline output doesn't match model output"

        # create bert model for sequence classification
        model = transformers.BertForSequenceClassification(config)
        with warnings.catch_warnings(record=True) as record:
            # expected to throw warning
            # badge uses hidden states as embedding by default but model uses pooler_output
            # results in mismatch between pipeline used by badge and model
            badge = Badge(model)
            # check infered classification type
            assert not badge.is_tok_cls, "Expected Sequence Classification but inferred Token Classification"
            # check expectation
            if not any((message in w.message.args[0]) for w in record):
                pytest.fail("Expected Warning: %s" % message)
        
        model = transformers.BertForTokenClassification(config)
        with warnings.catch_warnings(record=True) as record:
            # expects no warning
            # token classification model uses hidden states as embedding and simple linear
            # classifier on top, exactly matches badge pipeline
            badge = Badge(model)
            # check infered classification type
            assert badge.is_tok_cls, "Expected Token Classification but inferred Sequence Classification"
            # check expectation
            if any((message in w.message.args[0]) for w in record):
                pytest.fail("Unexpected Warning: %s" % message)


class TestExpectedGradientLength:
    """ Test cases for Expected Gradient Length """

    def test_goodfellow_grad_norm(self):
        """ Test Gradient Norm Computation using Goodfellow """        
        # create sample model with linear components
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 4),
            nn.Sigmoid(),
            nn.Linear(4, 2)
        )
        # some sample input data
        x = torch.rand(8, 2)
        y = (torch.rand(8) > 0.5).long()

        # attach gradient norm hooks
        grad_norm = GoodfellowGradientNorm(model)

        model.zero_grad()
        # compute gradient norm using goodfellow
        # use summation to avoid average over batches
        # this matches the computation with batch size 1
        F.cross_entropy(model(x), y, reduction='sum').backward()
        goodfellow_norm = grad_norm.compute()

        # compute norms manually to compare
        for i in range(x.size(0)):
            # pass through model
            model.zero_grad()
            F.cross_entropy(model(x[i:i+1]), y[i:i+1]).backward()
            # compute norm
            manual_norm = sum(
                (p.grad * p.grad).sum() 
                for n, p in model.named_parameters() if 'weight' in n
            )
            # make sure norms match
            assert torch.allclose(manual_norm, goodfellow_norm[i]), "%.04f != %.04f" % (manual_norm, goodfellow_norm[i])
    
    def test_goodfellow_grad_norm_multiple_backward_passes(self):
        """ Test multi-backward pass of Goodfellow Norm Computation """
        # create sample model with linear components
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 4),
            nn.Sigmoid(),
            nn.Linear(4, 2)
        )
        # some sample input data
        x = torch.rand(8, 2)

        # attach gradient norm hooks
        grad_norm = GoodfellowGradientNorm(model)

        # define reductions
        # use only summations again to match the
        # computations done with batch size set to 1
        # (compare `test_goodfellow_grad_norm`)
        reductions = [
            lambda t: t.sum(),
            lambda t: t.sin().sum(),
            lambda t: t.exp().sum()
        ]

        model.zero_grad()
        # multiple backward passes with different reductions
        y = model.forward(x)
        for fn in reductions:
            fn(y).backward(retain_graph=True)
        # compute gradient norm using goodfellow
        goodfellow_norm = grad_norm.compute()

        # compute norms manually to compare
        for i in range(x.size(0)):
            for j, fn in enumerate(reductions):
                # pass through model
                model.zero_grad()
                fn(model.forward(x[i:i+1])).backward()
                # compute norm
                manual_norm = sum(
                    (p.grad * p.grad).sum() 
                    for n, p in model.named_parameters() if 'weight' in n
                )
                # make sure norms match
                assert torch.allclose(manual_norm, goodfellow_norm[i, j]), "%.04f != %.04f" % (manual_norm, goodfellow_norm[i, j])
